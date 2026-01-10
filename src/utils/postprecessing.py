import torch
import pandas as pd
import numpy as np
from .metric import calculate_ef_from_areas

def generate_clinical_pairs(model, loader, device, save_path):
    """
    Runs inference to calculate LV areas, pairs ED/ES frames, 
    computes EF, and saves the results to a CSV file.
    
    Args:
        model (torch.nn.Module): Trained segmentation model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run inference on.
        save_path (str): Path to save the output CSV.
        
    Returns:
        pd.DataFrame: DataFrame containing paired clinical metrics.
    """
    model.eval()
    area_rows = []
    LV_LABEL = 1

    dev_type = device.type if hasattr(device, 'type') else str(device)
    with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
        for batch in loader:
            imgs = batch["image"].to(device)
            gts = batch["label"].to(device)
            cases, views, phases = batch["case"], batch["view"], batch["phase"]

            logits, _, _ = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for i in range(len(imgs)):
                gt_i = gts[i, 0].cpu().numpy()
                pr_i = preds[i].cpu().numpy()

                area_gt = float((gt_i == LV_LABEL).sum())
                area_pred = float((pr_i == LV_LABEL).sum())

                area_rows.append({
                    "case": cases[i],
                    "view": views[i],
                    "phase": phases[i],
                    "area_gt": area_gt,
                    "area_pred": area_pred
                })

    df_areas = pd.DataFrame(area_rows)
    
    # Pair ED/ES to calculate EF
    pairs = []
    for (case, view), grp in df_areas.groupby(["case", "view"]):
        phs = set(grp["phase"].tolist())
        if not {"ED", "ES"}.issubset(phs):
            continue

        ed = grp[grp["phase"] == "ED"].iloc[0]
        es = grp[grp["phase"] == "ES"].iloc[0]

        ef_ref = calculate_ef_from_areas(ed["area_gt"], es["area_gt"])
        ef_pred = calculate_ef_from_areas(ed["area_pred"], es["area_pred"])

        pairs.append({
            "case": case, "view": view,
            "ED_ref": ed["area_gt"], "ES_ref": es["area_gt"], "EF_ref": ef_ref,
            "ED_pred": ed["area_pred"], "ES_pred": es["area_pred"], "EF_pred": ef_pred,
            "EF_error": ef_pred - ef_ref,
            "EF_abs_err": abs(ef_pred - ef_ref)
        })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(save_path, index=False)
    return df_pairs

def get_representative_cases(df_metrics, n=3):
    """
    Identifies Best, Median, and Worst cases based on LV Dice score.
    
    Args:
        df_metrics (pd.DataFrame): DataFrame containing metrics per case.
        n (int): Number of cases to retrieve for each category.
        
    Returns:
        dict: Dictionary containing list of metadata for 'best', 'median', and 'worst' cases.
    """
    sorted_df = df_metrics.sort_values(by="dice_LV")
    
    worst = sorted_df.head(n).to_dict('records')
    best = sorted_df.tail(n).to_dict('records')
    
    mid_idx = len(sorted_df) // 2
    median = sorted_df.iloc[mid_idx:mid_idx+1].to_dict('records')
    
    return {"best": best, "median": median, "worst": worst}