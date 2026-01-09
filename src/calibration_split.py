import os
import sys
import argparse
import yaml
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# Add root to path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.metric import calculate_ef_from_areas

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_nifti(path):
    return nib.load(path).get_fdata()

def calculate_patient_ef(pid, data_nii_dir):
    """
    Calculates the Ejection Fraction (EF) for a given patient.

    Estimates EF by averaging calculations from 2CH and 4CH views using
    ground truth masks. Area-based approximation is used as a proxy for volume
    in this context, aligning with the provided metric utilities.

    Args:
        pid (str): Patient ID.
        data_nii_dir (str): Directory containing patient NIfTI files.

    Returns:
        float: Calculated mean EF. Returns 50.0 if calculation fails.
    """
    efs = []
    
    for view in ["2CH", "4CH"]:
        pdir = os.path.join(data_nii_dir, pid)
        
        # Paths for GT (ED and ES)
        lbl_ed_path = os.path.join(pdir, f"{pid}_{view}_ED_gt.nii.gz")
        lbl_es_path = os.path.join(pdir, f"{pid}_{view}_ES_gt.nii.gz")
        
        if os.path.exists(lbl_ed_path) and os.path.exists(lbl_es_path):
            lbl_ed = load_nifti(lbl_ed_path)
            lbl_es = load_nifti(lbl_es_path)
            
            # LV label is 1
            LV_LABEL = 1
            
            area_ed = (lbl_ed == LV_LABEL).sum()
            area_es = (lbl_es == LV_LABEL).sum()
            
            ef = calculate_ef_from_areas(area_ed, area_es)
            efs.append(ef)
            
    if not efs:
        print(f"Warning: Could not calculate EF for {pid}. Defaulting to 50.0")
        return 50.0
        
    return np.mean(efs)

def split_validation_set(config_path):
    """
    Splits the validation set into calibration and test subsets.

    The split is stratified by Ejection Fraction (EF) to ensure balanced representation
    of clinical severity in both subsets.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    cfg = load_config(config_path)
    
    split_dir = cfg['data']['split_dir']
    nii_dir = cfg['data']['nifti_dir']
    
    val_list_path = os.path.join(split_dir, "subgroup_validation.txt")
    
    if not os.path.exists(val_list_path):
        print(f"Error: Validation list not found at {val_list_path}")
        return

    with open(val_list_path, 'r') as f:
        val_ids = [l.strip() for l in f if l.strip()]
        
    print(f"Found {len(val_ids)} validation patients.")
    
    print("Calculating EF for stratification...")
    ef_values = []
    for pid in val_ids:
        ef = calculate_patient_ef(pid, nii_dir)
        ef_values.append(ef)
        
    # Bin EF values for stratification: [0, 35, 45, 55, 100]
    bins = [0, 35, 45, 55, 100]
    ef_bins = np.digitize(ef_values, bins)
    
    print("Splitting into Calibration (50%) and Test (50%)...")
    
    try:
        cal_ids, test_ids = train_test_split(
            val_ids, test_size=0.5, stratify=ef_bins, random_state=cfg['training']['seed']
        )
    except ValueError as e:
        print(f"Warning: Stratification failed ({e}). Falling back to simple random split.")
        cal_ids, test_ids = train_test_split(
            val_ids, test_size=0.5, random_state=cfg['training']['seed']
        )
        
    print(f"Calibration set: {len(cal_ids)} patients")
    print(f"Test set: {len(test_ids)} patients")
    
    # Save
    cal_path = os.path.join(split_dir, "subgroup_val_calibration.txt")
    test_path = os.path.join(split_dir, "subgroup_val_test.txt")
    
    # Write files
    # Note: We need to write to the split_dir
    # Check if we can write there clearly.
    try:
        with open(cal_path, 'w') as f:
            f.write('\n'.join(cal_ids))
            
        with open(test_path, 'w') as f:
            f.write('\n'.join(test_ids))
            
        print(f"Saved to:\n  {cal_path}\n  {test_path}")
    except OSError as e:
        print(f"Error writing split files: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    split_validation_set(args.config)
