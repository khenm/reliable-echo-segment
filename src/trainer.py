import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from src.utils.logging import get_logger

logger = get_logger()

class Trainer:
    """
    Handles the training, validation, and testing of the segmentation model.
    """
    def __init__(self, model, loaders, cfg, device):
        """
        Initializes the Trainer.
        
        Args:
            model (torch.nn.Module): The model to train.
            loaders (tuple): (train_loader, val_loader, test_loader).
            cfg (dict): Configuration dictionary.
            device (torch.device): Device to run on.
        """
        self.model = model
        self.ld_tr, self.ld_va, self.ld_ts = loaders
        self.cfg = cfg
        self.device = device
        self.num_classes = cfg['data']['num_classes']
        self.ckpt_path = cfg['training']['ckpt_save_path']
        
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metr_dice_val = DiceMetric(include_background=False, reduction="mean")
        self.opt = torch.optim.AdamW(model.parameters(), 
                                     lr=cfg['training']['lr'], 
                                     weight_decay=cfg['training']['weight_decay'])
        
        dev_type = device.type if hasattr(device, 'type') else str(device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))

    def train(self):
        """
        Executes the training loop with early stopping.
        """
        epochs = self.cfg['training']['epochs']
        patience = self.cfg['training']['patience']
        best_metric = 0.0
        wait = 0
        stop_ep = epochs

        start_tr = time.time()
        
        for ep in range(1, epochs + 1):
            self.model.train()
            run_loss = 0.0
            
            for batch in self.ld_tr:
                self.opt.zero_grad(set_to_none=True)
                imgs = batch["image"].to(self.device)
                labs = batch["label"].to(self.device)
                
                dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
                
                with torch.amp.autocast(device_type=dev_type):
                    logits = self.model(imgs)
                    loss = self.loss_fn(logits, labs)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                run_loss += loss.item()

            val_dice = self._validate()
            logger.info(f"E{ep:03d} trainLoss={run_loss/len(self.ld_tr):.4f} valDice={val_dice:.4f}")

            if val_dice > best_metric:
                logger.info(f"Saving checkpoint to {self.ckpt_path}...")
                torch.save(self.model.state_dict(), self.ckpt_path)
                logger.info("Checkpoint saved.")
                best_metric = val_dice
                wait = 0
            else:
                wait += 1
            
            if wait >= patience:
                logger.info("⏹ Early stop")
                stop_ep = ep
                break
        
        train_time = time.time() - start_tr
        logger.info(f"🏁 Finished at epoch {stop_ep} (best={best_metric:.4f}) in {train_time/60:.1f} min")

    def _validate(self):
        """
        Runs validation pass and returns the mean Dice score.
        """
        self.model.eval()
        self.metr_dice_val.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for vb in self.ld_va:
                v_img = vb["image"].to(self.device)
                v_lab = vb["label"].to(self.device)

                v_logits = self.model(v_img)
                v_pred_labels = torch.argmax(v_logits, dim=1)

                if v_lab.ndim == 4 and v_lab.shape[1] == 1:
                    v_gt_labels = v_lab[:, 0].long()
                else:
                    raise RuntimeError(f"Unexpected val GT shape {v_lab.shape}")

                v_y_pred = F.one_hot(v_pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                v_y_true = F.one_hot(v_gt_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

                self.metr_dice_val(v_y_pred, v_y_true)
        
        return float(self.metr_dice_val.aggregate().cpu())

    def evaluate_test(self):
        """
        Evaluates the best model on the test set, computing Dice and HD95 metrics.
        
        Returns:
            pd.DataFrame: DataFrame containing per-sample metrics.
        """
        logger.info(f"Loading best checkpoint from: {self.ckpt_path}")
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval()

        dice_metric = DiceMetric(include_background=False, reduction="none")
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

        records = []
        
        with torch.no_grad():
            for batch in self.ld_ts:
                imgs = batch["image"].to(self.device)
                gts = batch["label"].to(self.device)
                cases, views, phases = batch["case"], batch["view"], batch["phase"]

                logits = self.model(imgs)
                pred_labels = torch.argmax(logits, dim=1)

                if gts.ndim == 4 and gts.shape[1] == 1:
                    gt_labels = gts[:, 0].long()
                else:
                    gt_labels = gts.long()

                y_pred_all = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                y_true_all = F.one_hot(gt_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

                for i in range(len(imgs)):
                    y_p, y_t = y_pred_all[i:i+1], y_true_all[i:i+1]
                    
                    dice_metric.reset()
                    hd95_metric.reset()
                    dice_metric(y_p, y_t)
                    hd95_metric(y_p, y_t)
                    
                    dice_vals = dice_metric.aggregate().cpu().numpy().flatten()
                    hd95_vals = hd95_metric.aggregate().cpu().numpy().flatten()

                    records.append({
                        "case": cases[i], "view": views[i], "phase": phases[i],
                        "dice_LV": dice_vals[0], "dice_MYO": dice_vals[1], "dice_LA": dice_vals[2],
                        "hd95_LV": hd95_vals[0], "hd95_MYO": hd95_vals[1], "hd95_LA": hd95_vals[2],
                    })

        df = pd.DataFrame(records)
        df.to_csv(self.cfg['training']['test_metrics_csv'], index=False)
        logger.info(f"Saved metrics to {self.cfg['training']['test_metrics_csv']}")
        return df