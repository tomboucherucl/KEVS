import os
import nibabel as nib
import numpy as np
from scipy.stats import gaussian_kde
from util_new.erode_sat import binary_erode_from_exterior_2d

def kevs_predictions(scan_dir, umamba_prediction_dir):
    scan_list = os.listdir(scan_dir)
    for scan in sorted(scan_list):
        # Load image data
        scan_nii = nib.load(os.path.join(scan_dir, scan))
        scan_nii_data = scan_nii.get_fdata()
        
        # Load prediction data
        umamba_pred_nii = nib.load(os.path.join(umamba_prediction_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        umamba_pred_data = umamba_pred_nii.get_fdata()
        
        # Load SAT and abdominal cavity predictions
        umamba_sat_mask = (umamba_pred_data == 118)
        umamba_abd_cav_pred = (umamba_pred_data == 120)
        
        # Find the middle slice of L3 along the z-axis
        middle_slice = int(np.median(np.where(umamba_pred_data == 29)[2]))
        image_slice = scan_nii_data[:, :, middle_slice]
        sat_mask_slice = umamba_sat_mask[:, :, middle_slice]
        
        # Erode SAT mask
        eroded_sat_mask_slice = binary_erode_from_exterior_2d(sat_mask_slice, structuring_element = np.ones((3, 3)))
        eroded_sat_mask = np.zeros_like(image_slice, dtype=np.int16)
        eroded_sat_mask[sat_mask_slice > 0] = 118
        eroded_sat_intensities = image_slice[eroded_sat_mask_slice]
        
        # Fit GKDE
        kernel = gaussian_kde(eroded_sat_intensities)
        abd_cav_intensities = scan_nii_data[umamba_abd_cav_pred]
        kde_abd_values = kernel(abd_cav_intensities)
        
        # Set thresholding percentiles
        percentiles = [0, 5, 10, 15, 20, 25]
        for p in percentiles:
            
            # Calculate lower threshold of probability density values.
            lower_threshold = np.percentile(kde_abd_values, p)
            abd_cav_filtered_mask = np.zeros_like(umamba_abd_cav_pred)
            abd_cav_filtered_mask[umamba_abd_cav_pred] = kde_abd_values > lower_threshold
            
            # Set output directory (create it if it doesn't already exist)
            output_dir = f'./data/vat/predictions/KEVS/pd_{p}'
            os.makedirs(output_dir, exist_ok=True)

            # Save prediction to output dir
            kevs_pred = nib.Nifti1Image(abd_cav_filtered_mask.astype(np.int16), scan_nii.affine)
            nib.save(kevs_pred, os.path.join(output_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
    
    
    

