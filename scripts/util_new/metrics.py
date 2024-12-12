import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def extract_boundaries_2d(mask):
    """Extract the boundaries of a binary mask."""
    # Structuring element for erosion
    struct = np.ones((3, 3), dtype=bool)
    
    # Erode the mask to find the boundaries
    eroded_mask = binary_erosion(mask, structure=struct)
    
    # The boundary is the original mask minus the eroded mask
    boundary = mask & ~eroded_mask
    
    return boundary

def nsd_score_2d(mask_gt, mask_pred, tolerance, voxel_spacing=(1.5, 1.5)):
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    # Extract boundaries
    boundary_gt = extract_boundaries_2d(mask_gt)
    boundary_pred = extract_boundaries_2d(mask_pred)

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
    
    if boundary_length_gt + boundary_length_pred == 0:
        return 1
    
    if boundary_length_gt*boundary_length_pred == 0:
        return 0

    # Calculate NSD
    nsd = (np.sum(intersection_gt) + np.sum(intersection_pred)) / (boundary_length_gt + boundary_length_pred)
    
    return nsd

def extract_boundaries_3d(mask):
    """Extract the boundaries of a binary mask."""
    struct = np.ones((3, 3, 3), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=struct)
    boundary = mask & ~eroded_mask
    return boundary
    
def nsd_score_3d(mask_gt, mask_pred, tolerance, voxel_spacing=(1.5, 1.5, 1.5)):
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
    boundary_gt = extract_boundaries_3d(mask_gt)
    boundary_pred = extract_boundaries_3d(mask_pred)

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