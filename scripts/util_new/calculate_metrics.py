import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

from metrics import nsd_score, dice_score, sensitivity_score, precision_score


def get_pred_dirs(base_dirs):
    """Gets a list of prediction directories."""
    pred_dirs = []
    for base_dir in base_dirs:
        print(f"Processing base directory: {base_dir}")  # Debug print
        for root, _, files in os.walk(base_dir):
            print(f"  Checking directory: {root}")  # Debug print
            if any(f.endswith('.nii.gz') for f in files):
                print(f"    Found .nii.gz files in: {root}")  # Debug print
                pred_dirs.append(root)
    return pred_dirs

def calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list):
    results = {}  # Store results in a dictionary

    for pred_dir in tqdm(sorted(pred_dirs_list)):
        metrics = {'dice': [], 'nsd': [], 'precision': [], 'recall': [],
                   'dice_bounded': [], 'nsd_bounded': [], 'precision_bounded': [], 'recall_bounded': []}
        
        for ground_truth_filename in sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')]):
            mask_path = os.path.join(ground_truth_dir, ground_truth_filename)
            pred_path = os.path.join(pred_dir, ground_truth_filename)
            full_pred_path = os.path.join(full_pred_dir, ground_truth_filename)

            mask_numpy = nib.load(mask_path).get_fdata()
            pred_numpy = nib.load(pred_path).get_fdata()
            full_pred_numpy = nib.load(full_pred_path).get_fdata()

            # --- Full Cavity Metrics ---
            metrics['dice'].append(dice_score(pred_numpy.astype(int), mask_numpy.astype(int)))
            metrics['nsd'].append(nsd_score(mask_numpy, pred_numpy, tolerance=2))
            metrics['precision'].append(precision_score(mask_numpy, pred_numpy))
            metrics['recall'].append(sensitivity_score(mask_numpy, pred_numpy))

            # --- Bounded Region Metrics ---
            lower_z_bound = np.max(np.where(full_pred_numpy == 26)[2])
            upper_z_bound = np.min(np.where(full_pred_numpy == 32)[2])

            mask_bounded = mask_numpy.copy()
            pred_bounded = pred_numpy.copy()
            
            mask_bounded[:, :, :lower_z_bound] = False
            mask_bounded[:, :, upper_z_bound + 1:] = False
            pred_bounded[:, :, :lower_z_bound] = False
            pred_bounded[:, :, upper_z_bound + 1:] = False

            metrics['dice_bounded'].append(dice_score(pred_bounded.astype(int), mask_bounded.astype(int)))
            metrics['nsd_bounded'].append(nsd_score(mask_bounded, pred_bounded, tolerance=2))
            metrics['precision_bounded'].append(precision_score(mask_bounded, pred_bounded))
            metrics['recall_bounded'].append(sensitivity_score(mask_bounded, pred_bounded))

        results[pred_dir] = metrics  # Store metrics for the current directory

    return results

#Get path to current script and dir
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

ground_truth_dir = os.path.join(script_dir, "..", "..", "data", "vat", "ground_truth")
full_pred_dir = os.path.join(script_dir, "..", "..", "data", "umamba_predictions")
base_dirs = [os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "KEVS"), os.path.join(script_dir, "..","..", "data", "vat", "predictions", "thresholding"), os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "TotalSegmentator")]

pred_dirs_list = get_pred_dirs(base_dirs)

metrics_results = calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list)


    

