import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

def extract_boundaries(mask):
    """Extract the boundaries of a binary mask."""
    struct = np.ones((3, 3, 3), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=struct)
    boundary = mask & ~eroded_mask
    return boundary
    
def nsd_score(mask_gt, mask_pred, tolerance, voxel_spacing=(1.5, 1.5, 1.5)):
    """
    Calculate the Normalized Surface Distance (NSD) using the boundary-based metric.
    
    Parameters:
    - mask_gt: numpy array of ground truth binary mask.
    - mask_pred: numpy array of predicted binary mask.
    - tolerance: the tolerance value (τ).
    - voxel_spacing: tuple indicating the spacing of voxels in each dimension.
    
    Returns:
    - nsd: Normalized Surface Distance.
    """
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    # Extract boundaries
    boundary_gt = extract_boundaries(mask_gt)
    boundary_pred = extract_boundaries(mask_pred)

    # Compute distance transforms
    distmap_gt = distance_transform_edt(~boundary_gt, sampling=voxel_spacing)
    distmap_pred = distance_transform_edt(~boundary_pred, sampling=voxel_spacing)

    # Determine border regions within the specified tolerance
    border_region_gt = distmap_gt <= tolerance
    border_region_pred = distmap_pred <= tolerance

    # Compute intersection areas
    intersection_gt = np.logical_and(boundary_gt, border_region_pred)
    intersection_pred = np.logical_and(boundary_pred, border_region_gt)

    # Calculate the lengths of the boundaries
    boundary_length_gt = np.sum(boundary_gt)
    boundary_length_pred = np.sum(boundary_pred)

    # Calculate NSD
    nsd = (np.sum(intersection_gt) + np.sum(intersection_pred)) / (boundary_length_gt + boundary_length_pred)
    
    return nsd

def sensitivity_score(truth, prediction):
    """
    Calculate sensitivity (recall) for 3D images.
    """
    tp = np.sum((truth == 1) & (prediction == 1))
    fn = np.sum((truth == 1) & (prediction == 0))
    if tp + fn == 0:
        return 1  # Indeterminate sensitivity
    return tp / (tp + fn)

def precision_score(truth, prediction):
    """
    Calculate specificity for 3D images.
    """
    tp = np.sum((truth == 1) & (prediction == 1))
    fp = np.sum((truth == 0) & (prediction == 1))
    if tp + fp == 0:
        return 1  # Indeterminate specificity
    return tp / (tp + fp)

def dice_score(pred_matrix, mask_matrix):
    numerator = np.sum(pred_matrix*mask_matrix)
    return (2*numerator)/(np.sum(pred_matrix) + np.sum(mask_matrix))

def get_pred_dirs(base_dirs):
    """Gets a list of prediction directories."""
    pred_dirs = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            if any(f.endswith('.nii.gz') for f in files):
                pred_dirs.append(root)
    return pred_dirs

def calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list):
    results = {}  # Store results in a dictionary

    for pred_dir in tqdm(sorted(pred_dirs_list)):
        metrics = {'dice': [], 'nsd': [], 'precision': [], 'recall': [],
                   'dice_bounded': [], 'nsd_bounded': [], 'precision_bounded': [], 'recall_bounded': []}
        
        print(pred_dir)
        
        for ground_truth_filename in tqdm(sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')])):
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

ground_truth_dir = os.path.join(script_dir, "..", "data", "vat", "ground_truth")
full_pred_dir = os.path.join(script_dir, "..", "data", "umamba_predictions")
base_dirs = [os.path.join(script_dir, "..", "data", "vat", "predictions", "KEVS"), os.path.join(script_dir, "..", "data", "vat", "predictions", "thresholding"), os.path.join(script_dir, "..", "data", "vat", "predictions", "TotalSegmentator")]
pred_dirs_list = get_pred_dirs(base_dirs)

metrics_results = calculate_metrics(ground_truth_dir, full_pred_dir, pred_dirs_list)

print(metrics_results)

            
    

