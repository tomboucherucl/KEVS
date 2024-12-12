import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import wilcoxon
from tqdm import tqdm
import pandas as pd
import json
import itertools

from metrics import nsd_score_2d, dice_score, sensitivity_score, precision_score


def calculate_metrics_slices_pairwise(full_pred_dir, ground_truth_dir, pred_dirs_file, output_prefix="metrics_results"):
    # Load MRN associations from JSON file
    with open(pred_dirs_file, 'r') as f:
        pred_dirs = json.load(f)

    # Get all technique combinations
    techniques = sorted(list(pred_dirs.keys()))
    print(techniques)
    technique_combinations = list(itertools.combinations(techniques, 2))

    results = {}  # Initialize the results dictionary

    for technique1, technique2 in tqdm(technique_combinations, desc="Processing technique pairs"):
        print(f"Comparing techniques: {technique1} vs {technique2}")

        metrics = {
            f'{technique1}_dice': [], f'{technique1}_nsd': [], f'{technique1}_precision': [], f'{technique1}_recall': [],
            f'{technique2}_dice': [], f'{technique2}_nsd': [], f'{technique2}_precision': [], f'{technique2}_recall': []
        }

        # Determine the MRNs to use for comparison (MRNs of the technique NOT in the current pair)
        comparison_mrns = []
        for technique in techniques:
            if technique != technique1 and technique != technique2:
                comparison_mrns.extend(pred_dirs[technique]["mrns"])
        comparison_mrns = list(set(comparison_mrns))  # Remove duplicates

        if not comparison_mrns:
            print(f"    Warning: No comparison MRNs found for {technique1} vs {technique2}. Skipping this pair.")
            continue

        for ground_truth_filename in tqdm(sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz') and f[:8] in comparison_mrns]), desc=f"  Processing ground truth files ({technique1} vs {technique2})", leave=False):
            mask_path = os.path.join(ground_truth_dir, ground_truth_filename)
            full_pred_path = os.path.join(full_pred_dir, ground_truth_filename)

            try:
                mask_numpy = nib.load(mask_path).get_fdata()
                full_pred_numpy = nib.load(full_pred_path).get_fdata()
            except Exception as e:
                print(f"    Error loading file: {e}")
                continue

            num_axial_slices = mask_numpy.shape[-1]

            # --- Bounded Region ---
            lower_z_bound = np.max(np.where(full_pred_numpy == 26)[2])
            upper_z_bound = np.min(np.where(full_pred_numpy == 32)[2])

            mask_numpy[:, :, :lower_z_bound] = False
            mask_numpy[:, :, upper_z_bound + 1:] = False

            # Get paths for both techniques
            pred_paths = {
                technique1: [os.path.join(pred_dir_path, ground_truth_filename) for pred_dir_path in pred_dirs[technique1]["pred_dir_paths"]],
                technique2: [os.path.join(pred_dir_path, ground_truth_filename) for pred_dir_path in pred_dirs[technique2]["pred_dir_paths"]]
            }

            for technique, paths in pred_paths.items():
                for pred_path in paths:
                    if not os.path.exists(pred_path):
                        print(f"    Warning: Prediction file not found for technique {technique}: {pred_path}")
                        continue

                    try:
                        pred_numpy = nib.load(pred_path).get_fdata()
                    except Exception as e:
                        print(f"    Error loading prediction file for technique {technique}: {e}")
                        continue

                    # Apply bounding
                    pred_numpy[:, :, :lower_z_bound] = False
                    pred_numpy[:, :, upper_z_bound + 1:] = False

                    for slice_idx in range(num_axial_slices):
                        mask_slice = mask_numpy[:, :, slice_idx]
                        pred_slice = pred_numpy[:, :, slice_idx]

                        #if np.sum(mask_slice) > 0:
                        metrics[f'{technique}_dice'].append(dice_score(pred_slice.astype(int), mask_slice.astype(int)))
                        metrics[f'{technique}_nsd'].append(nsd_score_2d(mask_slice, pred_slice, tolerance=2))
                        metrics[f'{technique}_precision'].append(precision_score(mask_slice, pred_slice))
                        metrics[f'{technique}_recall'].append(sensitivity_score(mask_slice, pred_slice))

        # Perform Wilcoxon test and store results
        wilcoxon_results = {}
        for metric_name in ['dice', 'nsd', 'precision', 'recall']:
            try:
                statistic, p_value = wilcoxon(metrics[f'{technique1}_{metric_name}'], metrics[f'{technique2}_{metric_name}'], alternative='greater')
                wilcoxon_results[metric_name] = {'statistic': statistic, 'p_value': p_value}
            except ValueError as e:
                print(f"    Warning: Could not perform Wilcoxon test for {metric_name} ({technique1} vs {technique2}): {e}")
                wilcoxon_results[metric_name] = {'statistic': np.nan, 'p_value': np.nan}

        # Create DataFrame for CSV output
        df = pd.DataFrame(index=['dice', 'nsd', 'precision', 'recall'])

        for metric_name in ['dice', 'nsd', 'precision', 'recall']:
            mean_t1 = np.mean(metrics[f'{technique1}_{metric_name}'])
            std_t1 = np.std(metrics[f'{technique1}_{metric_name}'])
            mean_t2 = np.mean(metrics[f'{technique2}_{metric_name}'])
            std_t2 = np.std(metrics[f'{technique2}_{metric_name}'])

            df.loc[metric_name, f'{technique1}'] = f"{mean_t1:.4f} +/- {std_t1:.4f}"
            df.loc[metric_name, f'{technique2}'] = f"{mean_t2:.4f} +/- {std_t2:.4f}"
            df.loc[metric_name, 'Wilcoxon_stat'] = wilcoxon_results[metric_name]['statistic']
            df.loc[metric_name, 'Wilcoxon_p_value'] = wilcoxon_results[metric_name]['p_value']

        # Write DataFrame to CSV file
        df.to_csv(f"{output_prefix}_{technique1}_vs_{technique2}.csv")

    return results

# --- Example usage ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

ground_truth_dir = os.path.join(script_dir, "..","..", "data", "vat", "ground_truth")
full_pred_dir = os.path.join(script_dir, "..", "..", "data", "umamba_predictions")  # Assuming you have this
pred_dirs_file = os.path.join(script_dir, "mrn_associations.json")  # Path to your JSON file

# Call the function
results = calculate_metrics_slices_pairwise(full_pred_dir, ground_truth_dir, pred_dirs_file, output_prefix="metrics_results_axial")


