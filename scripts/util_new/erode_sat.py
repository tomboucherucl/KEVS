import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes

def binary_erode_from_exterior_2d(mask_2d, structuring_element):
    """
    Erodes a 2D binary mask from the exterior until a certain portion of the
    original volume is reached.

    Args:
        mask_2d: The 2D binary mask.
        structuring_element: The structuring element for erosion.

    Returns:
        The eroded 2D binary mask.
    """
    mask_copy = np.copy(mask_2d)
    filled_mask = binary_fill_holes(mask_copy)
    inner_block = filled_mask & ~mask_copy
    orig_num_voxels = np.sum(mask_copy)
    num_voxels = orig_num_voxels
    while num_voxels > int(orig_num_voxels * 0.2):
        filled_mask = binary_erosion(filled_mask, structure=structuring_element)
        true_eroded_sat = filled_mask & ~inner_block
        num_voxels = np.sum(true_eroded_sat)
    return true_eroded_sat

