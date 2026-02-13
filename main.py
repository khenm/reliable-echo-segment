import yaml
from src.utils.config_loader import load_config
import src.datasets
import src.models
import src.losses
import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from src.utils.util_ import seed_everything, get_device, load_checkpoint
from src.registry import get_dataloaders, build_model, build_loss, get_tta_component_class
from src.trainer import Trainer
from src.utils.logging import get_logger
from src.models.temporal import TemporalConsistencyLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from src.utils.plot import (
    plot_metrics_summary, 
    plot_clinical_bland_altman, 
    plot_reliability_curves, 
    plot_ef_category_roc,
    plot_clinical_comparison,
    plot_conformal_segmentation,
    plot_martingale
)
from src.utils.metric import SkeletalError, R2Score, MAE, RMSE
from src.tta.engine import TTAEngine, SafeTTAEngine
from src.tta.auditor import SelfAuditor
import glob

# Setup logging (stdout only initially)
logger = get_logger()

def resolve_resume_path(resume_mode, runs_root, model_name, cfg_resume_path=None):
    """
    Resolves the checkpoint path for resuming based on the mode.
    
    Args:
        resume_mode (str|bool): 'auto', path string, or True (treated as 'auto').
        runs_root (str): Root directory for runs (e.g., 'runs').
        model_name (str): Name of the model (for directory scoping).
        cfg_resume_path (str | None): Explicit path from config.
        
    Returns:
        str | None: Path to the checkpoint to resume from, or None if fresh run.
    """
    # 1. Explicit path in config takes precedence
    if cfg_resume_path:
        if os.path.exists(cfg_resume_path):
             logger.info(f"Resuming from config path: {cfg_resume_path}")
             return cfg_resume_path
        else:
             logger.warning(f"Configured resume_path {cfg_resume_path} not found. Falling back to logic.")

    # 2. If mode is None or False, fresh run
    if not resume_mode:
        return None
        
    # 3. If mode is a specific path, verify and return
    if isinstance(resume_mode, str) and resume_mode != 'auto' and "/" in resume_mode:
        if os.path.exists(resume_mode):
            return resume_mode
        else:
            logger.warning(f"Explicit resume path {resume_mode} not found. Starting fresh.")
            return None
            
    # 4. Auto-Resume Logic
    model_runs_dir = os.path.join(runs_root, model_name)
    if not os.path.exists(model_runs_dir):
        logger.info(f"No previous runs found for {model_name}. Starting fresh.")
        return None
        
    # List all run directories, sorted by creation time (descending)
    runs = sorted([
        os.path.join(model_runs_dir, d) for d in os.listdir(model_runs_dir) 
        if os.path.isdir(os.path.join(model_runs_dir, d))
    ], key=os.path.getmtime, reverse=True)
    
    for run in runs:
        last_ckpt = os.path.join(run, "last.ckpt")
        if os.path.exists(last_ckpt):
            logger.info(f"Auto-Discovery: Found resume point at {last_ckpt}")
            return last_ckpt
            
    logger.info("Auto-Discovery: No 'last.ckpt' found in recent history. Starting fresh.")
    return None

def _setup_workspace(cfg, args_resume):
    model_name = cfg['model'].get('name', 'VAEUNet')
    
    # 1. Directory Setup
    runs_root = cfg['training'].get('save_dir', 'runs')
    os.makedirs(runs_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    ckpt_root = cfg['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)

    vault_dir = os.path.join(ckpt_root, model_name)
    os.makedirs(vault_dir, exist_ok=True)
    
    # Determine Resume State
    resume_mode = 'auto' if args_resume else cfg['training'].get('resume_mode', None)
    cfg_resume_path = cfg['training'].get('resume_path')
    
    resume_path = resolve_resume_path(resume_mode, runs_root, model_name, cfg_resume_path)
    
    if resume_path:
        run_dir = os.path.dirname(resume_path)
        logger.info(f"Resuming workspace: {run_dir}")
        cfg['training']['resume_path'] = resume_path
        
        run_dir = os.path.join(runs_root, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created new workspace for resume session: {run_dir}")
    else:
        run_dir = os.path.join(runs_root, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created new workspace: {run_dir}")
    
    cfg['training']['run_dir'] = run_dir
    cfg['training']['vault_dir'] = vault_dir
    cfg['training']['run_id'] = timestamp
    
    return run_dir, vault_dir, timestamp

def _setup_wandb(cfg, run_dir, timestamp):
    if cfg.get('wandb', {}).get('enable', True):
        import wandb
        project_name = cfg.get('wandb', {}).get('project', 'reliable-echo-segment')
        
        if cfg['model'].get('name'):
            run_name = f"{cfg['model']['name']}_{timestamp}"
        else:
            run_name = f"run_{timestamp}"
            
        wandb.init(
            project=project_name,
            config=cfg,
            name=run_name,
            dir=run_dir,
            resume="allow",
            id=cfg['training'].get('run_id')
        )

def run_init(cfg, args_resume):
    run_dir, vault_dir, timestamp = _setup_workspace(cfg, args_resume)
    
    # Setup Logging
    log_file = os.path.join(run_dir, ".log")
    _ = get_logger(log_file=log_file)
    
    _setup_wandb(cfg, run_dir, timestamp)
    
    # Dump Config
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
    
    logger.info("Initializing...")
    logger.info(f"Workspace: {run_dir}")
    logger.info(f"Vault: {vault_dir}")
    
    seed_everything(cfg['training']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Update paths
    cfg['training']['ckpt_save_path'] = os.path.join(run_dir, "last.ckpt")
    
    metrics_csv = cfg['training'].get('test_metrics_csv', 'test_metrics.csv')
    if "/" in metrics_csv: metrics_csv = os.path.basename(metrics_csv)
    cfg['training']['test_metrics_csv'] = os.path.join(run_dir, metrics_csv)
    
    clinical_csv = cfg['training'].get('clinical_pairs_csv', 'clinical_pairs.csv')
    if "/" in clinical_csv: clinical_csv = os.path.basename(clinical_csv)
    cfg['training']['clinical_pairs_csv'] = os.path.join(run_dir, clinical_csv)

    return device

def run_preprocess(cfg):
    logger.info("Checking data loading...")
    loaders = get_dataloaders(cfg)
    
    for i, loader in enumerate(loaders):
        if loader is None: continue
        name = ["Train", "Val", "Test"][i]
        for _ in tqdm(loader, desc=f"Verifying {name}", mininterval=1.0):
            pass
            
    logger.info("Data verification complete.")

def _load_model_for_inference(cfg, device):
    model_name = cfg['model']['name']
    
    if cfg['training'].get('resume_path'):
         ckpt_path = cfg['training']['resume_path']
         if not os.path.exists(ckpt_path):
             logger.warning(f"Explicit path {ckpt_path} not found. Falling back to auto-discovery.")
         else:
             logger.info(f"Using explicit checkpoint: {ckpt_path}")
             model = build_model(cfg, device)
             load_checkpoint(model, ckpt_path, device)
             model.eval()
             return model

    vault_dir = os.path.join(cfg['training'].get('checkpoint_dir', 'checkpoints'), model_name)
    best_ckpts = glob.glob(os.path.join(vault_dir, "*best.ckpt"))
    
    if best_ckpts:
        best_ckpts.sort(key=os.path.getmtime, reverse=True)
        ckpt_path = best_ckpts[0]
        logger.info(f"Found BEST checkpoint: {ckpt_path}")
    else:
        logger.info("No best checkpoint found. Searching for latest last.ckpt...")
        runs_root = cfg['training'].get('save_dir', 'runs')
        search_pattern = os.path.join(runs_root, model_name, "*", "last.ckpt")
        last_ckpts = glob.glob(search_pattern)
        
        if last_ckpts:
            last_ckpts.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = last_ckpts[0]
            logger.info(f"Found LATEST checkpoint: {ckpt_path}")
        else:
            logger.error(f"No checkpoints found for model {model_name} in {vault_dir} or {runs_root}.")
            return None
            
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint path determined but not found: {ckpt_path}")
        return None
        
    model = build_model(cfg, device)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    return model

def _get_criterions(cfg):
    model_name = cfg['model'].get('name', 'VAEUNet')
    is_regression = (model_name.lower() == "r2plus1d")
    criterions = {}
    weights = cfg.get('loss', {}).get('weights', {})

    if is_regression:
        if weights.get('ef', 0.0) > 0:
            criterions['ef'] = build_loss("DifferentiableEFLoss", pixel_spacing=1.0, weight=weights.get('ef'))
        
        if weights.get('reg', 0.0) > 0:
            criterions['reg'] = torch.nn.L1Loss()
             
        criterions['seg'] = DiceCELoss(sigmoid=True, 
                                       lambda_dice=weights.get('dice', 0.7), 
                                       lambda_ce=weights.get('ce', 0.3))
    elif model_name in ["UNet_2D", "unet_tcm"]:
        losses_cfg = cfg.get('losses', {})
        if losses_cfg.get('consistency', {}).get('enable'):
            criterions['consistency'] = build_loss(
                "ConsistencyLoss",
                pixel_spacing=losses_cfg['consistency'].get('pixel_spacing', 0.3),
                step_size=losses_cfg['consistency'].get('step_size', 0.3),
                detach_gradients=losses_cfg['consistency'].get('detach_gradients', True)
            )
        
        if losses_cfg.get('temporal', {}).get('enable'):
             criterions['temporal'] = TemporalConsistencyLoss()
             
        num_classes = cfg['data'].get('num_classes', 1)
        if num_classes == 1:
            criterions['seg'] = DiceCELoss(sigmoid=True)
        else:
            criterions['seg'] = DiceCELoss(to_onehot_y=True, softmax=True)
        criterions['ef_reg'] = torch.nn.MSELoss() 

    elif model_name == "dual_stream":
        criterions['ef'] = torch.nn.MSELoss()
        criterions['simpson'] = torch.nn.L1Loss()
        
        num_classes = cfg['data'].get('num_classes', 1)
        if num_classes == 1:
            criterions['seg'] = DiceCELoss(sigmoid=True)
        else:
            criterions['seg'] = DiceCELoss(to_onehot_y=True, softmax=True)

    elif model_name == "skeletal_tracker":
        criterions['skeletal'] = build_loss(
            "SkeletalLoss",
            smooth_weight=weights.get('smooth', 0.1),
            topology_weight=weights.get('topology', 1.0),
            skeletal_weight=weights.get('skeletal', 1.0)
        )
        
        if weights.get('ef', 0.0) > 0:
            criterions['ef'] = torch.nn.MSELoss()

    elif model_name == "segment_tracker":
        criterions['segmentation'] = build_loss(
            "WeakSegLoss",
            dice_weight=weights.get('dice', 1.0),
            ef_weight=weights.get('ef', 10.0),
            smooth_weight=weights.get('smooth', 1.0),
            contrast_weight=weights.get('contrast', 0.1)
        )

    elif model_name == "temporal_segment_tracker":
        criterions['segmentation'] = build_loss(
            "TemporalWeakSegLoss",
            dice_weight=weights.get('dice', 1.0),
            ef_weight=weights.get('ef', 10.0),
            volume_weight=weights.get('volume', 0.0),
            cycle_weight=weights.get('cycle', 0.5)
        )

        distill_cfg = cfg.get('loss', {}).get('distillation', {})
        if distill_cfg.get('enabled', False):
            criterions['distillation'] = build_loss(
                "PanEchoDistillation",
                student_dim=distill_cfg.get('student_dim', 256),
                temperature=distill_cfg.get('temperature', 1.0),
                alpha=distill_cfg.get('alpha', 0.5)
            )
    
    else:
        criterions['dice'] = DiceCELoss(to_onehot_y=True, softmax=True)
        kl_weight = weights.get('kl', 1e-4)
        criterions['kl'] = build_loss("KLLoss", weight=kl_weight)
    
    return criterions

def _get_metrics(cfg):
    model_name = cfg['model'].get('name', 'VAEUNet')
    is_regression = (model_name.lower() == "r2plus1d")
    num_classes = cfg['data'].get('num_classes', 1)
    include_bg = (num_classes == 1)
    
    metrics = {}
    if is_regression or model_name == "skeletal_tracker":
        metrics['mae'] = MAE(reduction="mean")
        metrics['rmse'] = RMSE(reduction="mean")
        metrics['r2'] = R2Score()
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")
        if model_name == "skeletal_tracker":
            metrics['skeletal'] = SkeletalError()
    elif model_name in ["segment_tracker", "temporal_segment_tracker"]:
        metrics['mae'] = MAE(reduction="mean")     # EF
        metrics['rmse'] = RMSE(reduction="mean")   # EF
        metrics['r2'] = R2Score()                  # EF
        
        # EDV
        metrics['mae_edv'] = MAE(reduction="mean")
        metrics['rmse_edv'] = RMSE(reduction="mean")
        metrics['r2_edv'] = R2Score()

        # ESV
        metrics['mae_esv'] = MAE(reduction="mean")
        metrics['rmse_esv'] = RMSE(reduction="mean")
        metrics['r2_esv'] = R2Score()
        
        metrics['dice'] = DiceMetric(include_background=True, reduction="mean")
    else:
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")
    return metrics

def run_train(cfg, device):
    logger.info("Starting Training...")
    loaders = get_dataloaders(cfg)
    model = build_model(cfg, device)
    
    if cfg.get('wandb', {}).get('enable', True):
        import wandb
        if wandb.run is not None:
            wandb.watch(model, log="gradients", log_freq=100)
    
    criterions = _get_criterions(cfg)
    metrics = _get_metrics(cfg)

    trainer = Trainer(model, loaders, cfg, device, criterions=criterions, metrics=metrics)
    trainer.train()

def run_eval(cfg, device):
    logger.info("Starting Evaluation...")
    loaders = get_dataloaders(cfg)
    _, _, ld_ts = loaders
    
    model = _load_model_for_inference(cfg, device)
    if model is None:
        logger.error("Skipping evaluation due to missing checkpoint.")
        return
    
    metrics = _get_metrics(cfg)
    trainer = Trainer(model, loaders, cfg, device, metrics=metrics)
    trainer.evaluate_test()

    logger.info("Generating visualization examples...")
    try:
        samples = trainer.get_examples(num_examples=3)
        for s in samples:
            save_name = os.path.join(cfg['training']['run_dir'], f"vis_{s['fname']}_f{s['frame_idx']}.png")
            plot_clinical_comparison(s, save_path=save_name)
    except Exception as e:
        logger.error(f"Failed to plot visualization examples: {e}")

def run_tta(cfg, device):
    logger.info("Starting Test-Time Adaptation (TTA)...")
    
    original_bs = cfg['training'].get('batch_size')
    cfg['training']['batch_size'] = 1
    loaders = get_dataloaders(cfg)
    cfg['training']['batch_size'] = original_bs
    _, _, ld_ts = loaders
    
    model = _load_model_for_inference(cfg, device)
    if model is None: return
    
    tta_cfg = cfg.get('tta', {})
    engine = TTAEngine(
        model, 
        lr=float(tta_cfg.get('lr', 1e-4)), 
        n_augments=int(tta_cfg.get('n_augments', 4)),
        steps=int(tta_cfg.get('steps', 1)),
        optimizer_name=tta_cfg.get('optimizer', 'SGD')
    )
    engine.model.to(device)
    
    results = []
    logger.info(f"Running TTA on {len(ld_ts)} videos...")
    
    for batch in tqdm(ld_ts, desc="TTA Inference"):
        engine.reset()
        video = batch["video"].to(device)
        target = batch["target"].to(device)
        case = batch["case"][0]
        
        pred = engine.forward_and_adapt(video)
        results.append({
            "FileName": case,
            "Actual_EF": target.item(),
            "Predicted_EF": pred.item()
        })
        
    df_results = pd.DataFrame(results)
    save_path = cfg['training']['clinical_pairs_csv'].replace(".csv", "_tta.csv")
    df_results.to_csv(save_path, index=False)
    logger.info(f"TTA Results saved to {save_path}")
    
    mae = (df_results["Actual_EF"] - df_results["Predicted_EF"]).abs().mean()
    logger.info(f"TTA MAE: {mae:.2f}")

def run_plot(cfg):
    logger.info("Generating plots...")
    metrics_path = cfg['training']['test_metrics_csv']
    ef_path = cfg['training']['clinical_pairs_csv']
    
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        plot_metrics_summary(df_metrics, save_path=os.path.join(cfg['training']['run_dir'], "plot_metrics_summary.png"))
    
    if os.path.exists(ef_path):
        df_ef = pd.read_csv(ef_path)
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path) 
            plot_reliability_curves(df_metrics, df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_reliability.png"))
        
        plot_clinical_bland_altman(df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_bland_altman.png"))
        plot_ef_category_roc(df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_ef_roc.png"))
        logger.info("Generated clinical plots.")

def run_profile(cfg, device):
    logger.info("Starting Latent Profiling...")
    model = _load_model_for_inference(cfg, device)
    if model is None: return
    
    loaders = get_dataloaders(cfg)
    ld_tr, _, _ = loaders
    
    latent_dim = cfg['model']['latent_dim']
    
    run_dir = cfg['training']['run_dir']
    save_path = os.path.join(run_dir, "latent_profile.joblib")
    profiler.save(save_path)
    logger.info(f"Latent profile saved to {save_path}")

def _get_calibrator(cfg):
    model_name = cfg['model'].get('name', 'VAEUNet')
    is_dual = ("r2plus1d" in model_name.lower()) or (model_name in ["unet_tcm", "dual_stream"])
    
    if is_dual:
         logger.info(f"Initializing AuditCalibrator for Dual-Stream Model ({model_name})...")
         RegCal = get_tta_component_class("regression_calibrator")
         SegCal = get_tta_component_class("segmentation_calibrator")
         AuditCal = get_tta_component_class("audit_calibrator")
         return AuditCal({
             'regression': RegCal(alpha=0.05),
             'segmentation': SegCal(alpha=0.05)
         })
    elif model_name == "temporal_segment_tracker":
         logger.info(f"Initializing AuditCalibrator for {model_name}...")
         RegCal = get_tta_component_class("regression_calibrator")
         SegCal = get_tta_component_class("segmentation_calibrator")
         AuditCal = get_tta_component_class("audit_calibrator")
         return AuditCal({
             'regression': RegCal(alpha=0.05),
             'segmentation': SegCal(alpha=0.05)
         })
    elif model_name == "UNet_2D":
        logger.info(f"Initializing SegmentationCalibrator for {model_name}...")
        SegCal = get_tta_component_class("segmentation_calibrator")
        return SegCal(alpha=0.05)
    else:
        num_classes = cfg['data'].get('num_classes', 10)
        logger.info(f"Initializing ClassificationCalibrator for {model_name}...")
        ClassCal = get_tta_component_class("classification_calibrator")
        return ClassCal(alpha=0.05, num_classes=num_classes)

def run_safe_tta(cfg, device):
    logger.info("Starting Safe-TTA...")
    
    original_bs = cfg['training'].get('batch_size')
    cfg['training']['batch_size'] = 1
    loaders = get_dataloaders(cfg)
    cfg['training']['batch_size'] = original_bs
    
    _, ld_val, ld_ts = loaders
    
    model = _load_model_for_inference(cfg, device)
    if model is None: return
        
    # 1. Initialize Components
    feature_dim = cfg['model'].get('latent_dim', 512) 
    auditor = SelfAuditor(feature_dim=feature_dim)
    calibrator = _get_calibrator(cfg)
    
    safe_engine = SafeTTAEngine(
        model, 
        auditor=auditor, 
        calibrator=calibrator,
        lr=float(cfg.get('tta', {}).get('lr', 1e-4))
    )
    safe_engine.model.to(device)
    
    # 2. Calibration Phase (Offline)
    safe_engine.run_calibration(ld_val, device)

    # 3. Test Phase (Online)
    results = []
    logger.info(f"Running Safe-TTA on {len(ld_ts)} videos...")
    
    plots_dir = os.path.join(cfg['training']['run_dir'], "safe_tta_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for batch in tqdm(ld_ts, desc="Safe-TTA Inference"):
        safe_engine.reset()
        video = batch["video"].to(device)
        target = batch.get("target")
        if target is None: target = batch.get("target_ef")
        if target is None: target = batch.get("EF")
        
        if target is not None: target = target.to(device)
        case = batch["case"][0] 
        
        output, q_used, collapsed, audit_stats = safe_engine.predict_step(video)
        
        # Plot Martingale Dynamics
        if isinstance(audit_stats, dict) and 'martingale' in audit_stats:
            m_vals = audit_stats['martingale']
            p_vals = audit_stats.get('p_value', [0.5]*len(m_vals))
            if not isinstance(m_vals, list): m_vals = [m_vals]
            if not isinstance(p_vals, list): p_vals = [p_vals]
            
            plot_martingale(
                m_vals, 
                p_vals, 
                case_name=str(case),
                save_path=os.path.join(plots_dir, f"martingale_{case}.png")
            )
        
        if collapsed:
            results.append({
                "FileName": case,
                "Status": "COLLAPSED",
                "Q_Val": q_used
            })
            continue

        if isinstance(output, dict):
            reg_res = output.get('regression')
            pred_ef_interval = (0.0, 0.0)
            pred_ef_mean = 0.0
            
            if isinstance(reg_res, dict) and 'intervals' in reg_res:
                intervals = reg_res['intervals']
                predictions = reg_res['predictions']
                pred_ef_interval = intervals[0] if intervals else (0.0, 0.0)
                pred_ef_mean = predictions[0] if predictions else 0.0
            elif isinstance(reg_res, list):
                pred_ef_interval = reg_res[0] if reg_res else (0.0, 0.0)
                pred_ef_mean = sum(pred_ef_interval) / 2
                
            results.append({
                "FileName": case,
                "Actual_EF": target[0].item() if (target is not None and target.numel() > 0) else -1,
                "Predicted_EF": pred_ef_mean,
                "Predicted_EF_Low": pred_ef_interval[0],
                "Predicted_EF_High": pred_ef_interval[1],
                "Q_Val": q_used,
                "Status": "OK"
            })
            
            seg_res = output.get('segmentation')
            if seg_res is not None:
                core_mask, shadow_mask = seg_res
                T_dim = 2 if video.ndim == 5 else 1
                mid_frame = video.shape[T_dim] // 2 
                case_str = str(case)
                save_name = os.path.join(cfg['training']['run_dir'], f"conformal_seg_{case_str}_f{mid_frame}.png")
                
                gt_mask = batch.get("label")
                if gt_mask is not None: gt_mask = gt_mask.to(device)

                plot_conformal_segmentation(
                    video=video, 
                    core_mask=core_mask, 
                    shadow_mask=shadow_mask, 
                    target_mask=gt_mask,
                    frame_idx=mid_frame,
                    save_path=save_name
                )
            
        elif isinstance(output, tuple):
             results.append({
                "FileName": case,
                "Status": "OK (Segmentation)",
                "Q_Val": q_used
            })

        elif isinstance(output, list):
            for i, p_set in enumerate(output):
                 results.append({
                    "FileName": case,
                    "Actual_Class": target.view(-1)[i].item() if (target is not None and target.numel() > i) else -1,
                    "Prediction_Set": str(p_set),
                    "Set_Size": len(p_set),
                    "Q_Val": q_used,
                    "Status": "OK"
                })
            
    df_results = pd.DataFrame(results)
    save_path = os.path.join(cfg['training']['run_dir'], "safe_tta_results.csv")
    df_results.to_csv(save_path, index=False)
    logger.info(f"Safe-TTA Results saved to {save_path}")

def main():
    """
    Main entry point for the Reliable Echo Segmentation Pipeline.
    Parses arguments and orchestrates the execution of pipeline steps.
    """
    parser = argparse.ArgumentParser(description="Reliable Echo Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--init", action="store_true", help="Run initialization check")
    parser.add_argument("--preprocess", action="store_true", help="Run data loading verification")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--eval", action="store_true", help="Run evaluation (metrics + EF)")
    parser.add_argument("--rcps", action="store_true", help="Run RCPS calibration and evaluation")
    parser.add_argument("--tta", action="store_true", help="Run Test-Time Adaptation")
    parser.add_argument("--safe_tta", action="store_true", help="Run Safe-TTA (Audited Conformal)")
    parser.add_argument("--adaptive", action="store_true", help="Run Adaptive Calibration")
    parser.add_argument("--profile", action="store_true", help="Run Latent Profiling")
    parser.add_argument("--plot", action="store_true", help="Generate all plots")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override wandb enable status based on flag
    if 'wandb' not in cfg: cfg['wandb'] = {}
    cfg['wandb']['enable'] = args.wandb

    run_all = args.all or not (args.init or args.preprocess or args.train or args.eval or args.plot or args.rcps or args.profile or args.adaptive or args.tta or args.safe_tta)

    device = run_init(cfg, args.resume)
    
    if args.resume and 'resume_path' not in cfg['training']:
        logger.warning("Resume requested but no previous run found. Starting fresh (unless specific path provided later).")

    if run_all or args.preprocess:
        run_preprocess(cfg)
    
    if run_all or args.train:
        run_train(cfg, device)

    if run_all or args.profile:
        run_profile(cfg, device)
        
    if run_all or args.tta:
        run_tta(cfg, device)

    if run_all or args.safe_tta:
        run_safe_tta(cfg, device)
        
    if run_all or args.eval:
        run_eval(cfg, device)
        
    if run_all or args.plot:
        run_plot(cfg)

if __name__ == "__main__":
    main()