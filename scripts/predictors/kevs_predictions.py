import os
import nibabel as nib
import numpy as np
from scipy.stats import gaussian_kde
from util_new.erode_sat import binary_erode_from_exterior_2d

def kevs_predictions(image, mask, image_dir, umamba_prediction_dir):
    # Load image data
    image_nii = nib.load(os.path.join(image_dir, image))
    image_nii_data = image_nii.get_fdata()
    
    # Load prediction data
    umamba_pred_nii = nib.load(os.path.join(umamba_prediction_dir, mask))
    umamba_pred_data = umamba_pred_nii.get_fdata()
    
    umamba_pred_binary = np.zeros_like(umamba_pred_data, dtype=np.int16)
    umamba_pred_binary[umamba_pred_data > 0] = 1
    
    umamba_sat_mask = (umamba_pred_data == 118)
    umamba_abd_cav_pred = (umamba_pred_data == 120)
    
    # Find the middle slice of L3 along the z-axis
    middle_slice = int(np.median(np.where(umamba_pred_data == 29)[2]))
    image_slice = image_nii_data[:, :, middle_slice]
    sat_mask_slice = umamba_sat_mask[:, :, middle_slice]
    
    eroded_sat_mask_slice = binary_erode_from_exterior_2d(sat_mask_slice, structuring_element = np.ones((3, 3)))
    eroded_sat_mask = np.zeros_like(image_slice, dtype=np.int16)
    eroded_sat_mask[sat_mask_slice > 0] = 118
    eroded_sat_intensities = image_slice[eroded_sat_mask_slice]
    
    kernel = gaussian_kde(eroded_sat_intensities)
    abd_cav_intensities = image_nii_data[umamba_abd_cav_pred]
    kde_abd_values = kernel(abd_cav_intensities)
    
    percentiles = [0, 5, 10, 15, 20, 25]  # Percentiles for thresholding

    for p in percentiles:
        lower_threshold = np.percentile(kde_abd_values, p)
        abd_cav_filtered_mask = np.zeros_like(umamba_abd_cav_pred)
        abd_cav_filtered_mask[umamba_abd_cav_pred] = kde_abd_values > lower_threshold

        output_dir = f'./data/vat/predictions/KEVS/pd_{p}'  # Dynamic output directory
        os.makedirs(output_dir, exist_ok=True)

        kevs_pred = nib.Nifti1Image(abd_cav_filtered_mask.astype(np.int16), image_nii.affine)
        nib.save(kevs_pred, os.path.join(output_dir, image.replace('_0000.nii.gz', '.nii.gz')))
    
    
    

