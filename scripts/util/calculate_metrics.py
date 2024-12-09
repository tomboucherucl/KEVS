import os
import numpy as np
import nibabel as nib
import pandas as pd

from scipy.stats import wilcoxon
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

pred_dirs_list = ['KEVS_combined_0', 'KEVS_separated_0', 'KEVS_combined_5', 'KEVS_separated_5', 'KEVS_combined_10', 'KEVS_separated_10',
                  'KEVS_combined_15', 'KEVS_separated_15', 'KEVS_combined_20', 'KEVS_separated_20', 'KEVS_combined_25', 'KEVS_separated_25', 
                  'TotalSegmentator_predictions', 'pred_190_30', 'pred_195_45', 'pred_200_10', 'pred_200_20', 'pred_250_50']

ground_truth_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/Edited_VAT_resampled'
kevs_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS'
thresholding_pred= '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_200_10'
ts_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/TotalSegmentator_predictions'

full_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_preds'

dice_list_kevs = []
dice_list_ts = []
dice_list_thresh = []

dice_ensamble = []

def extract_boundaries(mask):
    """Extract the boundaries of a binary mask."""
    # Structuring element for erosion
    struct = np.ones((3, 3, 3), dtype=bool)
    
    # Erode the mask to find the boundaries
    eroded_mask = binary_erosion(mask, structure=struct)
    
    # The boundary is the original mask minus the eroded mask
    boundary = mask & ~eroded_mask
    
    return boundary

def absolute_vol_difference(mask_gt, pred_gt):
    mask_gt = mask_gt.astype(int)
    pred_gt = pred_gt.astype(int)
    
    numerator = np.abs(np.sum(mask_gt)-np.sum(pred_gt))*100
    denomenator = np.sum(mask_gt)
    
    return numerator/denomenator
    
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

data = []

for dir in tqdm(pred_dirs_list):
    dir_path = f'/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/{dir}'
    dice_list = []
    nsd_list = []
    prec_list = []
    recall_list = []
    avd_list = []
    
    for ground_truth in sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')]):     
        patient_id = ground_truth[:8]
           
        mask_nii = nib.load(os.path.join(ground_truth_dir, ground_truth))
        mask_numpy = mask_nii.get_fdata()
        
        pred_nii = nib.load(os.path.join(dir_path, ground_truth))
        pred_numpy = pred_nii.get_fdata()
        
        full_pred_nii = nib.load(os.path.join(full_pred_dir, ground_truth))
        full_pred_numpy = full_pred_nii.get_fdata()
        # Set bounds
        lower_z_bound = np.max(np.where(full_pred_numpy == 26)[2])
        upper_z_bound = np.min(np.where(full_pred_numpy == 32)[2])
        
        mask_numpy[:, :, :lower_z_bound] = False
        mask_numpy[:, :, upper_z_bound + 1:] = False
        pred_numpy[:, :, :lower_z_bound] = False
        pred_numpy[:, :, upper_z_bound + 1:] = False
        
        dc = dice_score(pred_numpy.astype(int), mask_numpy.astype(int))
        dice_list.append(dc)
        
        nsd = nsd_score(mask_numpy, pred_numpy, tolerance=2)
        nsd_list.append(nsd)
        
        precision = precision_score(mask_numpy, pred_numpy)
        prec_list.append(precision)
        
        recall = sensitivity_score(mask_numpy, pred_numpy)
        recall_list.append(recall)
        
        avd = absolute_vol_difference(mask_numpy, pred_numpy)
        avd_list.append(avd)
        
    row = {
        'Method': dir,
        'Dice Coefficient': f"{np.mean(dice_list):.4f} +/- {np.std(dice_list):.4f}",
        'NSD Score': f"{np.mean(nsd_list):.4f} +/- {np.std(nsd_list):.4f}",
        'Precision score': f"{np.mean(prec_list):.4f} +/- {np.std(prec_list):.4f}",
        'Recall score': f"{np.mean(recall_list):.4f} +/- {np.std(recall_list):.4f}",
        'Absolute volume difference': f"{np.mean(avd_list):.4f} +/- {np.std(avd_list):.4f}",
        }
    data.append(row)
    
    print(f"{dir}")
    print(f"Dice Coefficient: {np.mean(dice_list):.4f} +/- {np.std(dice_list):.4f}")
    print(f"NSD Score: {np.mean(nsd_list):.4f} +/- {np.std(nsd_list):.4f}")
    print(f"Prec Score: {np.mean(prec_list):.4f} +/- {np.std(prec_list):.4f}")
    print(f"Recall Score: {np.mean(recall_list):.4f} +/- {np.std(recall_list):.4f}")
    print(f"AVD Score: {np.mean(avd_list):.4f} +/- {np.std(avd_list):.4f}")
    
    
    # Convert the data list into a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('VAT_metrics_full_cavity.csv', index=False)

    
""" print(f"Average KEVS over dataset {np.mean(dice_list_kevs)} +/- {np.std(dice_list_kevs)}")
print(f"Average TS over dataset {np.mean(dice_list_ts)} +/- {np.std(dice_list_ts)}")
print(f"Average thresh over dataset {np.mean(dice_list_thresh)} +/- {np.std(dice_list_thresh)}")
print(f"Average ensamble over dataset {np.mean(dice_ensamble)} +/- {np.std(dice_ensamble)}")

stat_thresh, p_value_thresh = wilcoxon(dice_list_kevs, dice_list_thresh, alternative='two-sided')

print(f'Statistic: {stat_thresh}')
print(f'P-value: {p_value_thresh}')

# Check the result
if p_value_thresh < 0.05:
    print("The result is statistically significant for threshold. kevs is likely greater than thresh.")
else:
    print("The result is not statistically significant.")
    
stat_ts, p_value_ts = wilcoxon(dice_list_kevs, dice_list_ts, alternative='greater')

print(f'Statistic: {stat_ts}')
print(f'P-value: {p_value_ts}')

# Check the result
if p_value_ts < 0.05:
    print("The result is statistically significant for threshold. kevs is likely greater than ts.")
else:
    print("The result is not statistically significant.")
    
print(dice_list_kevs)
print(dice_list_thresh)
print(dice_list_ts) """