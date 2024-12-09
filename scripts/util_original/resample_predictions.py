import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

def resample_to_spacing_pytorch(image_data, original_affine, new_spacing, interpolation_mode='trilinear'):
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

niftis_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_niftis'
masks_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/Edited VAT'
masks_resampled_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/Edited_VAT_resampled'

os.makedirs(masks_resampled_dir, exist_ok=True)
new_spacing = (1.5, 1.5, 1.5)

sorted_masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])

for mask in tqdm(sorted_masks, total = len([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])):
    # Load the image and mask
    mask_nii = nib.load(os.path.join(masks_dir, mask))
    mask_voxel_spacing = mask_nii.header.get_zooms()
    
    print(mask_voxel_spacing)
    
    if mask_voxel_spacing == (1.5,1.5,1.5):
        print(f"No need to resample this one!")
        name_for_image = mask[5:]
        nib.save(mask_nii, os.path.join(masks_resampled_dir, name_for_image))
        continue
    else:
        name_for_image = mask[5:]
        nifti_nii = nib.load(os.path.join(niftis_dir, name_for_image.replace('.nii.gz', '_0000.nii.gz')))

        # Resample the image using trilinear interpolation
        resampled_image_data, new_affine = resample_to_spacing_pytorch(
            nifti_nii.get_fdata(), 
            nifti_nii.affine, 
            new_spacing, 
            interpolation_mode='trilinear'
        )
        
        # Resample the mask using nearest neighbor interpolation
        resampled_mask_data, _ = resample_to_spacing_pytorch(
            mask_nii.get_fdata(), 
            mask_nii.affine, 
            new_spacing, 
            interpolation_mode='nearest'
        )

        # Create new Nifti images with the resampled data and the new affine matrix
        resampled_nifti_mask = nib.Nifti1Image(resampled_mask_data, new_affine)  # Using same affine to ensure alignment
        
        # Save the resampled image and mask
        nib.save(resampled_nifti_mask, os.path.join(masks_resampled_dir, name_for_image))
    
    
