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

def calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list, output_file="metrics_results.txt"):
    results = {}

    for pred_dir in tqdm(sorted(pred_dirs_list), desc="Processing prediction directories"):
        metrics = {'dice': [], 'nsd': [], 'precision': [], 'recall': [],
                   'dice_bounded': [], 'nsd_bounded': [], 'precision_bounded': [], 'recall_bounded': []}

        print(f"Processing prediction directory: {pred_dir}")

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

        # Calculate mean and standard deviation
        summary_metrics = {}
        for metric_name, metric_values in metrics.items():
            summary_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values)
            }

        results[pred_dir] = summary_metrics

    # Write results to a text file
    with open(output_file, "w") as f:
        for pred_dir, metrics in results.items():
            f.write(f"Results for {pred_dir}:\n")
            for metric_name, stats in metrics.items():
                f.write(f"  {metric_name}:\n")
                f.write(f"    Mean: {stats['mean']:.4f}\n")
                f.write(f"    Std: {stats['std']:.4f}\n")
            f.write("\n")  # Add an extra newline for separation

    return results

#Get path to current script and dir
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

ground_truth_dir = os.path.join(script_dir, "..", "..", "data", "vat", "ground_truth")
full_pred_dir = os.path.join(script_dir, "..", "..", "data", "umamba_predictions")
base_dirs = [os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "TotalSegmentator")]
#os.path.join(script_dir, "..", "..", "data", "vat", "predictions", "KEVS"), os.path.join(script_dir, "..","..", "data", "vat", "predictions", "thresholding"), 

pred_dirs_list = get_pred_dirs(base_dirs)

metrics_results = calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list, output_file="my_custom_results.txt")

# (Optional) Print the results to the console as well
for pred_dir, metrics in metrics_results.items():
    print(f"Results for {pred_dir}:")
    for metric_name, stats in metrics.items():
        print(f"  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")


    

