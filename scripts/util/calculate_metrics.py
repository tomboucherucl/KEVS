import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

from metrics import nsd_score_3d, dice_score, sensitivity_score, precision_score

def get_pred_dirs(base_dirs):
    """Gets a list of prediction directories."""
    pred_dirs = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            if any(f.endswith('.nii.gz') for f in files):
                pred_dirs.append(root)
    return pred_dirs

def calculate_metrics_full_kevs(ground_truth_dir, full_pred_dir, pred_dirs_list, output_prefix="metrics_results"):
    results = {}

    for pred_dir in tqdm(sorted(pred_dirs_list), desc="Processing prediction directories"):
        # Extract technique name from the last part of the path
        technique_name = os.path.basename(pred_dir)  # Get the last part of the path

        metrics = {'dice': [], 'nsd': [], 'precision': [], 'recall': [],
                   'dice_bounded': [], 'nsd_bounded': [], 'precision_bounded': [], 'recall_bounded': []}

        print(f"Processing prediction directory: {pred_dir} (Technique: {technique_name})")

        for ground_truth_filename in tqdm(sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')]), desc="  Processing ground truth files", leave=False):
            mask_path = os.path.join(ground_truth_dir, ground_truth_filename)
            pred_filename = ground_truth_filename
            pred_path = os.path.join(pred_dir, pred_filename)
            full_pred_filename = ground_truth_filename
            full_pred_path = os.path.join(full_pred_dir, full_pred_filename)

            if not os.path.exists(pred_path):
                print(f"    Warning: Prediction file not found: {pred_path}")
                continue

            try:
                mask_numpy = nib.load(mask_path).get_fdata()
                pred_numpy = nib.load(pred_path).get_fdata()
                full_pred_numpy = nib.load(full_pred_path).get_fdata()
            except Exception as e:
                print(f"    Error loading file: {e}")
                continue
            
            pred_numpy = (pred_numpy == 122).astype(int)
            # --- Full Cavity Metrics ---
            metrics['dice'].append(dice_score(pred_numpy.astype(int), mask_numpy.astype(int)))
            metrics['nsd'].append(nsd_score_3d(mask_numpy, pred_numpy, tolerance=2))
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
            metrics['nsd_bounded'].append(nsd_score_3d(mask_bounded, pred_bounded, tolerance=2))
            metrics['precision_bounded'].append(precision_score(mask_bounded, pred_bounded))
            metrics['recall_bounded'].append(sensitivity_score(mask_bounded, pred_bounded))

        # Calculate mean and standard deviation
        summary_metrics = {}
        for metric_name, metric_values in metrics.items():
            summary_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values)
            }
        results[technique_name] = summary_metrics  # Use technique name as the key
        print(summary_metrics)

    # Create DataFrames for CSV output
    metrics_names = ['dice', 'nsd', 'precision', 'recall']
    bounded_metrics_names = ['dice_bounded', 'nsd_bounded', 'precision_bounded', 'recall_bounded']

    df_unbounded = pd.DataFrame(index=metrics_names, columns=sorted(results.keys()))
    df_bounded = pd.DataFrame(index=bounded_metrics_names, columns=sorted(results.keys()))

    for technique_name, metrics in results.items():
        for metric_name in metrics_names:
            if metric_name in metrics:
                df_unbounded.loc[metric_name, technique_name] = f"{metrics[metric_name]['mean']:.4f} +/- {metrics[metric_name]['std']:.4f}"

        for metric_name in bounded_metrics_names:
            if metric_name in metrics:
                df_bounded.loc[metric_name, technique_name] = f"{metrics[metric_name]['mean']:.4f} +/- {metrics[metric_name]['std']:.4f}"

    # Write DataFrames to CSV files
    df_unbounded.to_csv(f"{output_prefix}_unbounded.csv")
    df_bounded.to_csv(f"{output_prefix}_bounded.csv")

    return results

def calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list, output_prefix="metrics_results"):
    results = {}

    for pred_dir in tqdm(sorted(pred_dirs_list), desc="Processing prediction directories"):
        # Extract technique name from the last part of the path
        technique_name = os.path.basename(pred_dir)  # Get the last part of the path

        metrics = {'dice': [], 'nsd': [], 'precision': [], 'recall': [],
                   'dice_bounded': [], 'nsd_bounded': [], 'precision_bounded': [], 'recall_bounded': []}

        print(f"Processing prediction directory: {pred_dir} (Technique: {technique_name})")

        for ground_truth_filename in tqdm(sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')]), desc="  Processing ground truth files", leave=False):
            mask_path = os.path.join(ground_truth_dir, ground_truth_filename)
            pred_filename = ground_truth_filename
            pred_path = os.path.join(pred_dir, pred_filename)
            full_pred_filename = ground_truth_filename
            full_pred_path = os.path.join(full_pred_dir, full_pred_filename)

            if not os.path.exists(pred_path):
                print(f"    Warning: Prediction file not found: {pred_path}")
                continue

            try:
                mask_numpy = nib.load(mask_path).get_fdata()
                pred_numpy = nib.load(pred_path).get_fdata()
                full_pred_numpy = nib.load(full_pred_path).get_fdata()
            except Exception as e:
                print(f"    Error loading file: {e}")
                continue

            # --- Full Cavity Metrics ---
            metrics['dice'].append(dice_score(pred_numpy.astype(int), mask_numpy.astype(int)))
            metrics['nsd'].append(nsd_score_3d(mask_numpy, pred_numpy, tolerance=2))
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
            metrics['nsd_bounded'].append(nsd_score_3d(mask_bounded, pred_bounded, tolerance=2))
            metrics['precision_bounded'].append(precision_score(mask_bounded, pred_bounded))
            metrics['recall_bounded'].append(sensitivity_score(mask_bounded, pred_bounded))

        # Calculate mean and standard deviation
        summary_metrics = {}
        for metric_name, metric_values in metrics.items():
            summary_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values)
            }
        results[technique_name] = summary_metrics  # Use technique name as the key
        print(summary_metrics)

    # Create DataFrames for CSV output
    metrics_names = ['dice', 'nsd', 'precision', 'recall']
    bounded_metrics_names = ['dice_bounded', 'nsd_bounded', 'precision_bounded', 'recall_bounded']

    df_unbounded = pd.DataFrame(index=metrics_names, columns=sorted(results.keys()))
    df_bounded = pd.DataFrame(index=bounded_metrics_names, columns=sorted(results.keys()))

    for technique_name, metrics in results.items():
        for metric_name in metrics_names:
            if metric_name in metrics:
                df_unbounded.loc[metric_name, technique_name] = f"{metrics[metric_name]['mean']:.4f} +/- {metrics[metric_name]['std']:.4f}"

        for metric_name in bounded_metrics_names:
            if metric_name in metrics:
                df_bounded.loc[metric_name, technique_name] = f"{metrics[metric_name]['mean']:.4f} +/- {metrics[metric_name]['std']:.4f}"

    # Write DataFrames to CSV files
    df_unbounded.to_csv(f"{output_prefix}_unbounded.csv")
    df_bounded.to_csv(f"{output_prefix}_bounded.csv")

    return results

#Get path to current script and dir
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

ground_truth_dir = os.path.join(script_dir, "..", "..", "data", "vat", "ground_truth")
full_pred_dir = os.path.join(script_dir, "..", "..", "data", "umamba_predictions")
base_dirs = [os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "KEVS")]#,os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "KEVS_full_volume")]#, os.path.join(script_dir, "..","..", "data", "vat", "predictions", "thresholding"), os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "TotalSegmentator")]
#o

#pred_dirs_list = get_pred_dirs(base_dirs)

metrics_results = calculate_metrics_full_kevs(ground_truth_dir, full_pred_dir, pred_dirs_list = get_pred_dirs([os.path.join(script_dir, "..", "..", "data", "KEVS_full_prediction")]))



    

