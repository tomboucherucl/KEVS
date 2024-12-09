import nibabel as nib
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

def resample_scan(image_data, original_affine, new_spacing, interpolation_mode='trilinear'):
    """
    General function to resample either images or masks with different interpolation modes.
    :param image_data: The image or mask data to be resampled.
    :param original_affine: The affine matrix of the original image/mask.
    :param new_spacing: Desired voxel spacing after resampling.
    :param interpolation_mode: 'trilinear' for images, 'nearest' for masks.
    :return: Resampled data and updated affine matrix.
    """
    # Extract the original voxel spacing from the affine matrix
    original_spacing = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))

    # Compute the new shape for the resampled data (based on new voxel spacing)
    new_shape = np.round(np.array(image_data.shape) * (original_spacing / new_spacing)).astype(int)

    # Convert the image data to a PyTorch tensor
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, D, H, W)

    # Perform the resampling
    resampled_tensor = F.interpolate(image_tensor, size=tuple(new_shape), mode=interpolation_mode)

    # Remove the added dimensions (1, 1) -> (D, H, W)
    resampled_image_data = resampled_tensor.squeeze().numpy()

    # Compute the new affine matrix
    new_affine = original_affine.copy()
    scaling_factors = 1/(np.array(original_spacing) / np.array(new_spacing))
    
    new_affine[:3, :3] = np.dot(original_affine[:3, :3], np.diag(scaling_factors))

    return resampled_image_data, new_affine

def resample_scans_dir(input_dir, output_dir, new_spacing, interpolation_mode='trilinear'):
    """
    General function to resample directories of either images or masks with different interpolation modes.
    :param input_dir: The directory of image or mask data to be resampled.
    :param output_dir: The directory of resampled image or mask data.
    :param new_spacing: Desired voxel spacing after resampling.
    :param interpolation_mode: 'trilinear' for images, 'nearest' for masks.
    :output: Directories of resampled data.
    """
    
    # Create output directory (if it does not already exist)
    os.makedirs(output_dir, exist_ok=True)
    
    for scan in sorted(input_dir):
        # Load the scan
        scan_nifti = nib.load(os.path.join(input_dir, scan))
        
        # Resampled the nifti
        resampled_scan, new_affine = resample_scan(scan_nifti.get_fdata(), 
                                                   scan_nifti.affine, 
                                                   new_spacing, 
                                                   interpolation_mode,
                                                   )
        
        # convert resampled scan back to nifti
        resample_scan_nifti = nib.Nifti1Image(resampled_scan, new_affine)
        
        # save resampled scan to output directory 
        nib.save(resample_scan_nifti, os.path.join(output_dir, scan.replace('.nii.gz', '_0000.nii.gz'))) #Change suffix for compatibility with nnU-Net (U-Mamba)


    
    
