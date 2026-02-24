import os
import yaml
import torch
import glob
from datetime import datetime
from src.utils.logging import get_logger
from src.utils.util_ import seed_everything, get_device, load_checkpoint
from src.registry import build_model, build_loss
from src.utils.dist import is_main_process
from src.models.temporal import TemporalConsistencyLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from src.utils.metric import SkeletalError, R2Score, MAE, RMSE

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
        
        # If resuming, we might want to keep the same run_dir logic or create a new one?
        # main.py logic was creating a NEW workspace even for resume, which is interesting.
        # "Created new workspace for resume session"
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
    # Only initialize wandb on the main process
    if not is_main_process():
        return

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
    # Re-configure logger to write to file
    global logger
    logger = get_logger(log_file=log_file)
    
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

def load_model_for_inference(cfg, device):
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

def get_criterions(cfg):
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

    elif model_name == "temporal_segment_tracker" or model_name == "cardiac_mamba":
        criterions['segmentation'] = build_loss(
            "TemporalWeakSegLoss",
            dice_weight=weights.get('dice', 1.0),
            volume_weight=weights.get('volume', 1.0),
            sv_weight=weights.get('sv', 1.0),
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

def get_metrics(cfg):
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
    elif model_name in ["segment_tracker", "temporal_segment_tracker", "cardiac_mamba"]:
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
        
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")
    else:
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")
    return metrics
