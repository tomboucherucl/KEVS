import nibabel as nib
import os
import numpy as np

def thresholding_predictions(scan_dir, saros_mask_dir):
    """
    Function to make thresholding VAT predictions on CT scans

    Args:
        scan_dir: The directory containing the scans.
        saros_mask_dir: The directory containing the SAROS masks.

    Returns:
        None. Saves processed images and KDE plots.
    """
    scan_list = os.listdir(scan_dir)
    for scan in sorted(scan_list):
        scan_nii = nib.load(os.path.join(scan_dir, scan))
        scan_nii_data = scan_nii.get_fdata()

        saros_mask_nii = nib.load(os.path.join(saros_mask_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        saros_mask_nii_data = saros_mask_nii.get_fdata()

        saros_abd_cav_mask = (saros_mask_nii_data == 3)

        # Define threshold ranges and corresponding output directory names
        threshold_ranges = {
            'range_minus190_minus30': (-190, -30),
            'range_minus200_minus10': (-200, -10),
            'range_minus200_minus20': (-200, -20),
            'range_minus250_minus50': (-250, -50),
            'range_minus195_minus45': (-195, -45),
        }

        base_output_dir = './data/vat/predictions/thresholding'

        for pred_name, (lower_thresh, upper_thresh) in threshold_ranges.items():
            # Create VAT mask based on threshold range
            vat_mask = (scan_nii_data >= lower_thresh) & (scan_nii_data <= upper_thresh) & saros_abd_cav_mask

            # Create NIfTI image from the mask
            pred_nii = nib.Nifti1Image(vat_mask.astype(np.int16), scan_nii.affine)

            # Create output directory if it doesn't exist
            output_dir = os.path.join(base_output_dir, pred_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save the NIfTI image
            output_filename = scan.replace('_0000.nii.gz', '.nii.gz')
            nib.save(pred_nii, os.path.join(output_dir, output_filename))