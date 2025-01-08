import os
import nibabel as nib
import torch
import numpy as np
from scipy.stats import gaussian_kde
from util.erode_sat import binary_erode_from_exterior_2d
import subprocess
import time
import os
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='kevs_prediction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def run_nnunet_prediction(scans_dir, output_dir, dataset_id, configuration, folds, trainer):
    command = [
        "nnUNetv2_predict",
        "-i", scans_dir,
        "-o", output_dir,
        "-d", dataset_id,
        "-c", configuration,
        "-f", folds,
        "-tr", trainer
    ]

    try:
        # Use subprocess.run to execute the command
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decode output as text
            env=os.environ,  # Pass the current environment variables
            check=True  # Raise an exception if the command fails
        )

        logging.info("Output: \n" + process.stdout)


    except subprocess.CalledProcessError as e:
        logging.error(f"Error running U-Mamba prediction: {e}")
        logging.error("Error Output: \n" + e.stderr)
        return None
    
def vat_gkde(scan_dir, umamba_prediction_dir, percentiles):
    scan_list = os.listdir(scan_dir)
    for scan in tqdm(sorted(f for f in scan_list if f.endswith('.nii.gz')), desc = 'Single slice KEVS inference'):
        logging.info(f"Beginning analysis of {scan.replace('_0000.nii.gz', '')}")
        # Load image data
        scan_nii = nib.load(os.path.join(scan_dir, scan))
        scan_nii_data = scan_nii.get_fdata()
        
        # Load prediction data
        umamba_pred_nii = nib.load(os.path.join(umamba_prediction_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        umamba_pred_data = umamba_pred_nii.get_fdata()
        
        combined_data = np.copy(umamba_pred_data)
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
        
        logging.info(f"There are {len(eroded_sat_intensities)} intensity values")
        
        # Fit GKDE
        before_fitting = time.time()
        kernel = gaussian_kde(eroded_sat_intensities)
        abd_cav_intensities = scan_nii_data[umamba_abd_cav_pred]
        kde_abd_values = kernel(abd_cav_intensities)
        fitting_time = time.time() - before_fitting
        
        logging.info(fitting_time)
            
        time_before_thresholding = time.time()
        for percentile in percentiles:
            logging.info(f"processing for percentile {percentile}")
            # Calculate lower threshold of probability density values.
            lower_threshold = np.percentile(kde_abd_values, percentile)
            
            abd_cav_filtered_mask = np.zeros_like(umamba_abd_cav_pred)
            abd_cav_filtered_mask[umamba_abd_cav_pred] = kde_abd_values > lower_threshold
            
            print(np.sum(abd_cav_filtered_mask))
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",  "data", "vat", "KEVS", f"pd_{percentile}" )
            os.makedirs(output_dir, exist_ok=True)        

            # Save prediction to output dir
            kevs_pred = nib.Nifti1Image(abd_cav_filtered_mask.astype(np.int16), scan_nii.affine)
            nib.save(kevs_pred, os.path.join(output_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        average_thresholding_time = (time.time() - time_before_thresholding)/len(percentiles)
        
    return fitting_time, average_thresholding_time
        
def vat_gkde_full_scan(scan_dir, umamba_prediction_dir, percentiles):
    # Set output directory (create it if it doesn't already exist)
    scan_list = os.listdir(scan_dir)
    logging.info(sorted(f for f in scan_list if f.endswith('.nii.gz')))
    for scan in tqdm(sorted(f for f in scan_list if f.endswith('.nii.gz')), desc = 'Bounded region KEVS prediction'):
        logging.info(f"Beginning analysis of {scan.replace('_0000.nii.gz', '')}")
        # Load image data
        scan_nii = nib.load(os.path.join(scan_dir, scan))
        scan_nii_data = scan_nii.get_fdata()
        
        # Load prediction data
        umamba_pred_nii = nib.load(os.path.join(umamba_prediction_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        umamba_pred_data = umamba_pred_nii.get_fdata()
        
        lower_z_bound = np.max(np.where(umamba_pred_data == 26)[2])
        upper_z_bound = np.min(np.where(umamba_pred_data == 32)[2])

        umamba_mask_bounded = umamba_pred_data.copy()
        
        # Load SAT and abdominal cavity predictions
        umamba_abd_cav_pred = (umamba_mask_bounded == 120)
        
        umamba_mask_bounded[:, :, :lower_z_bound] = False
        umamba_mask_bounded[:, :, upper_z_bound + 1:] = False
        
        umamba_sat_mask = (umamba_mask_bounded == 118)
        
        z_indices = np.where(umamba_sat_mask == True)[2]
        max_z = np.max(z_indices)
        min_z = np.min(z_indices)
        logging.info(f"min:{min_z}, max: {max_z}")
        eroded_sat_mask = np.zeros_like(scan_nii_data[:,:,:], dtype=np.int16)
        
        all_eroded_sat_intensities = []
        
        #print(f"There are {np.prod(umamba_sat_mask == True)} voxels in this mask")
        for slice_index in range(min_z, max_z+1):
            sat_mask_slice = umamba_sat_mask[:, :, slice_index]
            image_slice = scan_nii_data[:, :, slice_index]
            
            eroded_sat_mask_slice = binary_erode_from_exterior_2d(sat_mask_slice, structuring_element = np.ones((3, 3)))
            eroded_sat_mask[:,:,slice_index][sat_mask_slice > 0] = 118
            eroded_sat_intensities = image_slice[eroded_sat_mask_slice]
            all_eroded_sat_intensities.extend(eroded_sat_intensities)
            
        all_eroded_sat_intensities = np.array(all_eroded_sat_intensities)
        
        logging.info(f"There are {len(all_eroded_sat_intensities)} intensity values")
        
        # Fit GKDE
        logging.info(f"Fitting GKDE...")
        time_before = time.time()
        kernel = gaussian_kde(all_eroded_sat_intensities)
        abd_cav_intensities = scan_nii_data[umamba_abd_cav_pred]
        kde_abd_values = kernel(abd_cav_intensities)
        total_fitting_time = time.time() - time_before
        
        time_before_thresholding = time.time()
        for percentile in percentiles:
            # Calculate lower threshold of probability density values.
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",  "data", "vat", "KEVS", f"pd_{percentile}" )
            os.makedirs(output_dir, exist_ok=True)
            lower_threshold = np.percentile(kde_abd_values, percentile)
            abd_cav_filtered_mask = np.zeros_like(umamba_abd_cav_pred)
            abd_cav_filtered_mask[umamba_abd_cav_pred] = kde_abd_values > lower_threshold

            # Save prediction to output dir
            kevs_pred = nib.Nifti1Image(abd_cav_filtered_mask.astype(np.int16), scan_nii.affine)
            nib.save(kevs_pred, os.path.join(output_dir, scan.replace('_0000.nii.gz', '.nii.gz')))
        total_thresholding_time = time.time() - time_before_thresholding
        average_thresholding_time = total_thresholding_time/len(percentiles)
        
    return total_fitting_time, average_thresholding_time
            
def KEVS_prediction():
    scans_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "scans")
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "umamba_predictions")
    dataset_id = "200"
    configuration = "3d_fullres"
    folds = "all"
    trainer = "nnUNetTrainerUMambaBot"
    percentiles = [0, 5, 10, 15, 20, 25]
    
    #time_before_umamba = time.time()
    #run_nnunet_prediction(scans_directory, output_directory, dataset_id, configuration, folds, trainer)
    #time_after_umamba = time.time()
    #time_umamba = time_after_umamba - time_before_umamba
    
    # Set thresholding percentiles
    #logging.info(f"The total time for the U-Mamba predictions is {time_umamba} seconds")
    #logging.info(f"The average time for the U-Mamba predictions is {time_umamba/20} seconds")
    
    fitting_time, thresholding_time = vat_gkde(scans_directory, output_directory, percentiles)
    #full_fitting_time, full_thresholding_time = vat_gkde_full_scan(scans_directory, output_directory, percentiles)
    

    #logging.info(f"The total time for the bounded region KEVS prediction was {full_fitting_time + full_thresholding_time}s this is an average of {(full_fitting_time + full_thresholding_time)/20}s. Of this, {full_fitting_time}s was just for fitting the kernel.")
    #logging.info(f"The total time for the single slice KEVS was {fitting_time + thresholding_time} this is an average of {(fitting_time + thresholding_time)/98}. Of this, {fitting_time}s was just for fitting the kernel.")
        
        
    