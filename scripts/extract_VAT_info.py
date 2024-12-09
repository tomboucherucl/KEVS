import nibabel as nib
from nibabel.orientations import aff2axcodes, io_orientation, ornt_transform, apply_orientation
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from erode_sat import erode_sat, save_kde_plot

resampled_saros_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_SAROS_pred'
#image_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_niftis'

resampled_image_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_niftis_resampled_refined'
resampled_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_preds'

vat_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/KEVS_prediction'

os.makedirs(vat_dir, exist_ok=True)

labels = {
        #"background": 0,
        "spleen": 1,
        "kidney_right": 2,
        "kidney_left": 3,
        "gallbladder": 4,
        "liver": 5,
        "stomach": 6,
        "pancreas": 7,
        "adrenal_gland_right": 8,
        "adrenal_gland_left": 9,
        "lung_upper_lobe_left": 10,
        "lung_lower_lobe_left": 11,
        "lung_upper_lobe_right": 12,
        "lung_middle_lobe_right": 13,
        "lung_lower_lobe_right": 14,
        "esophagus": 15,
        "trachea": 16,
        "thyroid_gland": 17,
        "small_bowel": 18,
        "duodenum": 19,
        "colon": 20,
        "urinary_bladder": 21,
        "prostate": 22,
        "kidney_cyst_left": 23,
        "kidney_cyst_right": 24,
        "sacrum": 25,
        "vertebrae_S1": 26,
        "vertebrae_L5": 27,
        "vertebrae_L4": 28,
        "vertebrae_L3": 29,
        "vertebrae_L2": 30,
        "vertebrae_L1": 31,
        "vertebrae_T12": 32,
        "vertebrae_T11": 33,
        "vertebrae_T10": 34,
        "vertebrae_T9": 35,
        "vertebrae_T8": 36,
        "vertebrae_T7": 37,
        "vertebrae_T6": 38,
        "vertebrae_T5": 39,
        "vertebrae_T4": 40,
        "vertebrae_T3": 41,
        "vertebrae_T2": 42,
        "vertebrae_T1": 43,
        "vertebrae_C7": 44,
        "vertebrae_C6": 45,
        "vertebrae_C5": 46,
        "vertebrae_C4": 47,
        "vertebrae_C3": 48,
        "vertebrae_C2": 49,
        "vertebrae_C1": 50,
        "heart": 51,
        "aorta": 52,
        "pulmonary_vein": 53,
        "brachiocephalic_trunk": 54,
        "subclavian_artery_right": 55,
        "subclavian_artery_left": 56,
        "common_carotid_artery_right": 57,
        "common_carotid_artery_left": 58,
        "brachiocephalic_vein_left": 59,
        "brachiocephalic_vein_right": 60,
        "atrial_appendage_left": 61,
        "superior_vena_cava": 62,
        "inferior_vena_cava": 63,
        "portal_vein_and_splenic_vein": 64,
        "iliac_artery_left": 65,
        "iliac_artery_right": 66,
        "iliac_vena_left": 67,
        "iliac_vena_right": 68,
        "humerus_left": 69,
        "humerus_right": 70,
        "scapula_left": 71,
        "scapula_right": 72,
        "clavicula_left": 73,
        "clavicula_right": 74,
        "femur_left": 75,
        "femur_right": 76,
        "hip_left": 77,
        "hip_right": 78,
        "spinal_cord": 79,
        "gluteus_maximus_left": 80,
        "gluteus_maximus_right": 81,
        "gluteus_medius_left": 82,
        "gluteus_medius_right": 83,
        "gluteus_minimus_left": 84,
        "gluteus_minimus_right": 85,
        "autochthon_left": 86,
        "autochthon_right": 87,
        "iliopsoas_left": 88,
        "iliopsoas_right": 89,
        "brain": 90,
        "skull": 91,
        "rib_left_1": 92,
        "rib_left_2": 93,
        "rib_left_3": 94,
        "rib_left_4": 95,
        "rib_left_5": 96,
        "rib_left_6": 97,
        "rib_left_7": 98,
        "rib_left_8": 99,
        "rib_left_9": 100,
        "rib_left_10": 101,
        "rib_left_11": 102,
        "rib_left_12": 103,
        "rib_right_1": 104,
        "rib_right_2": 105,
        "rib_right_3": 106,
        "rib_right_4": 107,
        "rib_right_5": 108,
        "rib_right_6": 109,
        "rib_right_7": 110,
        "rib_right_8": 111,
        "rib_right_9": 112,
        "rib_right_10": 113,
        "rib_right_11": 114,
        "rib_right_12": 115,
        "sternum": 116,
        "costal_cartilages": 117,
        "Subcutaneous_Tissue": 118,
        "Muscle": 119,
        "Abdominal_Cavity":120,
        "Thoracic_Cavity":121
    }

def reorient_to_standard(nii, target_orientation=('L', 'P', 'S')):
    current_orientation = aff2axcodes(nii.affine)
    current_ornt = io_orientation(nii.affine)
    target_ornt = nib.orientations.axcodes2ornt(target_orientation)
    transform = ornt_transform(current_ornt, target_ornt)
    reoriented_data = apply_orientation(nii.get_fdata(), transform)
    reoriented_img = nib.Nifti1Image(reoriented_data, nii.affine)
    
    return reoriented_img, transform
# Loop through the prediction NIfTI files
def calculate_volume(image_nii, mask_nii, body_feature, labels, lower_threshold=None, upper_threshold=None):
    if body_feature == 'Subcutaneous_Tissue':
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()  # Extract the voxel data from the NIfTI object
        voxel_dimensions = mask_nii.header.get_zooms()
        voxel_volume = np.prod(voxel_dimensions)
        
        upper_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_T12"])[2]))
        lower_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_S1"])[2]))

        mask_data[:, :, :lower_vertical_indices] = False
        mask_data[:, :, upper_vertical_indices+1:] =  False
        
        # Filter for the specific body feature within the T12-S1 range
        body_feature_mask = (mask_data == labels[body_feature])
        
        intensity_mask = (image_data > lower_threshold) & (image_data < upper_threshold)
            
        # Combine the body feature mask with the intensity mask
        body_feature_mask = body_feature_mask & intensity_mask
        
        # Calculate the number of voxels for the specific body feature
        body_feature_num_voxels = np.sum(body_feature_mask)
        
        # Calculate the total volume in cubic centimeters (cc)
        total_volume = body_feature_num_voxels * voxel_volume / 1000  # Convert mm³ to cm³
        
    if body_feature == 'Muscle':
        mask_data = mask_nii.get_fdata()  # Extract the voxel data from the NIfTI object
        voxel_dimensions = mask_nii.header.get_zooms()
        voxel_volume = np.prod(voxel_dimensions)
        
        upper_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_T12"])[2]))
        lower_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_S1"])[2]))

        mask_data[:, :, :lower_vertical_indices] = False
        mask_data[:, :, upper_vertical_indices+1:] =  False
        
        # Filter for the specific body feature within the T12-S1 range
        body_feature_mask = (mask_data == labels[body_feature]) | (mask_data == 86) | (mask_data == 87) | (mask_data == 88) | (mask_data == 89)
        
        # Calculate the number of voxels for the specific body feature
        body_feature_num_voxels = np.sum(body_feature_mask)
        
        # Calculate the total volume in cubic centimeters (cc)
        total_volume = body_feature_num_voxels * voxel_volume / 1000  # Convert mm³ to cm³
        
    else:
        mask_data = mask_nii.get_fdata()  # Extract the voxel data from the NIfTI object
        voxel_dimensions = mask_nii.header.get_zooms()
        voxel_volume = np.prod(voxel_dimensions)
        
        upper_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_T12"])[2]))
        lower_vertical_indices = int(np.median(np.where(mask_data == labels["vertebrae_S1"])[2]))

        mask_data[:, :, :lower_vertical_indices] = False
        mask_data[:, :, upper_vertical_indices+1:] =  False
        
        # Filter for the specific body feature within the T12-S1 range
        body_feature_mask = (mask_data == labels[body_feature])
        
        # Calculate the number of voxels for the specific body feature
        body_feature_num_voxels = np.sum(body_feature_mask)
        
        # Calculate the total volume in cubic centimeters (cc)
        total_volume = body_feature_num_voxels * voxel_volume / 1000  # Convert mm³ to cm³
        
    return total_volume

def calculate_vertebrae_information(image_nii, mask_nii, vertebral_level, labels, body_feature = None, left_and_right=False, iliopsoas=False):
    image_data = image_nii.get_fdata() 
    mask_data = mask_nii.get_fdata() 
    
    voxel_dimensions = mask_nii.header.get_zooms() 
    voxel_area = voxel_dimensions[0] * voxel_dimensions[1]  # Axes 1 (y) and 2 (x)
    
    # Find the middle slice along the z-axis
    middle_slice = int(np.median(np.where(mask_data == labels[vertebral_level])[2]))
    
    if left_and_right == False:        
        # Get the mask for the body feature in the middle slice
        vertebral_slice_mask = (mask_data[:, :, middle_slice] == labels[body_feature])
        
        # Calculate the area based on the number of true values in the mask
        body_feature_area = np.sum(vertebral_slice_mask) * voxel_area /100
        
        # Apply the mask to the 2D slice of the image (on the middle slice)
        image_voxels = image_data[ :, :, middle_slice][vertebral_slice_mask]
        
        if len(image_voxels) > 0:
            avg_intensity = np.median(image_voxels)
            std_intensity = np.std(image_voxels)
        else:
            avg_intensity = None
            std_intensity = None
            
        return body_feature_area, avg_intensity, std_intensity
    
    elif left_and_right == True:
        if iliopsoas == True:
            vertebral_slice_mask = (mask_data[:, :, middle_slice] == 88) | (mask_data[:, :, middle_slice] == 89)
        elif iliopsoas == False:
            vertebral_slice_mask = (mask_data[:, :, middle_slice] == 86) | (mask_data[:, :, middle_slice] == 87)
            
        # Calculate the area based on the number of true values in the mask
        body_feature_area = np.sum(vertebral_slice_mask) * voxel_area/100
        
        # Apply the mask to the 2D slice of the image (on the middle slice)
        image_voxels = image_data[ :, :, middle_slice][vertebral_slice_mask]
        
        if len(image_voxels) > 0:
            avg_intensity = np.median(image_voxels)
            std_intensity = np.std(image_voxels)
        else:
            avg_intensity = None
            std_intensity = None
            
        return body_feature_area, avg_intensity, std_intensity

def calculate_vertebrae_intensity(image_nii, mask_nii, vertebral_level, labels):
    image_data = image_nii.get_fdata() 
    mask_data = mask_nii.get_fdata() 
    
    # Find the middle slice along the z-axis
    middle_slice = int(np.median(np.where(mask_data == labels[vertebral_level])[2]))
    
    # Get the mask for the body feature in the middle slice
    vertebral_slice_mask = (mask_data[:, :, middle_slice] == labels[vertebral_level])
    
    # Apply the mask to the 2D slice of the image (on the middle slice)
    image_voxels = image_data[ :, :, middle_slice][vertebral_slice_mask]
    
    if len(image_voxels) > 0:
        avg_intensity = np.median(image_voxels)
        std_intensity = np.std(image_voxels)
    else:
        avg_intensity = None
        std_intensity = None
        
    return avg_intensity, std_intensity

def vat_from_saros(resampled_image_dir, pred_dir):
    data = []
    for image in tqdm(sorted([f for f in os.listdir(resampled_image_dir) if f.endswith('.nii.gz')])):
        
        # Extract patient information
        patient_id = image[:7]
        series = image[18:26]
        
        # Load the corresponding image NIfTI file
        image_nifti = nib.load(os.path.join(resampled_image_dir, image))
        image_nifti_info = image_nifti.get_fdata()
        
        # Load the prediction NIfTI file
        pred_nifti = nib.load(os.path.join(pred_dir, image.replace('_0000.nii.gz', '.nii.gz')))
        pred_nifti_info = pred_nifti.get_fdata()

        # Get voxel volume information
        voxel_dimensions = pred_nifti.header.get_zooms()
        voxel_volume = np.prod(voxel_dimensions)
        
        # Create VAT mask where the prediction equals 3
        muscle_mask_in_pred = (pred_nifti_info == 119)
        vat_mask_in_pred = (pred_nifti_info == 120)
        
        # Extract the original image voxel values in the VAT mask region
        original_mask_muscle_voxels = image_nifti_info[muscle_mask_in_pred]
        original_mask_abd_voxels = image_nifti_info[vat_mask_in_pred]
        
        #calculate mean and std of intensity values of muscle voxels
        intensity_muscle_avg = np.mean(original_mask_muscle_voxels)
        intensity_muscle_std = np.std(original_mask_muscle_voxels)
        
        # Apply thresholding directly to the original image within the VAT region
        vat_mask_190_30_new = (original_mask_abd_voxels > -190) & (original_mask_abd_voxels < -30)
        vat_mask_200_10_new = (original_mask_abd_voxels > -200) & (original_mask_abd_voxels < -10)
        vat_mask_200_20_new = (original_mask_abd_voxels > -200) & (original_mask_abd_voxels < -20)
        vat_mask_250_50_new = (original_mask_abd_voxels > -250) & (original_mask_abd_voxels < -50)
        vat_mask_195_45_new = (original_mask_abd_voxels > -195) & (original_mask_abd_voxels < -45)

        # Create a new full mask for the entire volume with the same shape as the original image
        vat_mask_190_30_full = np.zeros_like(image_nifti_info, dtype=np.int16)
        vat_mask_200_10_full = np.zeros_like(image_nifti_info, dtype=np.int16)
        vat_mask_200_20_full = np.zeros_like(image_nifti_info, dtype=np.int16)
        vat_mask_250_50_full = np.zeros_like(image_nifti_info, dtype=np.int16)
        vat_mask_195_45_full = np.zeros_like(image_nifti_info, dtype=np.int16)
        
        # Create the new VAT mask depending on the specified threshold
        vat_mask_190_30_full[vat_mask_in_pred] = vat_mask_190_30_new.astype(np.int16)
        vat_mask_200_10_full[vat_mask_in_pred] = vat_mask_200_10_new.astype(np.int16)
        vat_mask_200_20_full[vat_mask_in_pred] = vat_mask_200_20_new.astype(np.int16)
        vat_mask_250_50_full[vat_mask_in_pred] = vat_mask_250_50_new.astype(np.int16)
        vat_mask_195_45_full[vat_mask_in_pred] = vat_mask_195_45_new.astype(np.int16)
        
        vat_mask_190_30_num_volume = np.sum(vat_mask_190_30_full) * voxel_volume/1000
        vat_mask_200_10_num_volume = np.sum(vat_mask_200_10_full) * voxel_volume/1000
        vat_mask_200_20_num_volume = np.sum(vat_mask_200_20_full) * voxel_volume/1000
        vat_mask_250_50_num_volume = np.sum(vat_mask_250_50_full) * voxel_volume/1000
        vat_mask_195_45_num_volume = np.sum(vat_mask_195_45_full) * voxel_volume/1000
        
        
        
        data.append([patient_id, series, vat_mask_190_30_num_volume, vat_mask_200_10_num_volume, vat_mask_200_20_num_volume, vat_mask_250_50_num_volume, vat_mask_195_45_num_volume, intensity_muscle_avg, intensity_muscle_std])
        
        
    df = pd.DataFrame(data, columns = ['Patient ID', 'Series ID', '(-190,-30) volume', '(-200,-10) volume', '(-200,-20) volume', '(-250,-50) volume', '(-195,-45) volume', 'average muscle intensity', 'std muscle intensity'])

    # Save the DataFrame to a CSV file
    df.to_csv('VAT.csv', index=False)
    
def calculate_vat_volume(mask, thresholds, voxel_volume):
    volumes = []
    for lower, upper in thresholds:
        vat_mask = (mask > lower) & (mask < upper)
        volume = np.sum(vat_mask) * voxel_volume / 1000
        volumes.append(volume)
    return volumes

def vat_from_saros_w_vertebrae(resampled_image_dir, resampled_pred_dir, resampled_saros_pred_dir, vat_dir, labels):
    vertebrae_list = ["vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1"]
    thresholds = [(-190, -30), (-200, -10), (-200, -20), (-250, -50), (-195, -45)]
    data = []
    
    print(f"Length of IPCAI nifti resampled refined: {len(os.listdir(resampled_image_dir))}")
    
    for image in tqdm(sorted([f for f in os.listdir(resampled_image_dir) if f.endswith('.nii.gz')])):
        # Extract patient information
        patient_id = image[:8]
        series_id = image[18:26]

        # Load image and prediction NIfTI files
        image_path = os.path.join(resampled_image_dir, image)
        image_nifti = nib.load(image_path)
        
        pred_path = os.path.join(resampled_pred_dir, image.replace('_0000.nii.gz', '.nii.gz'))
        pred_nifti = nib.load(pred_path)
        voxel_dimensions = pred_nifti.header.get_zooms() 
        voxel_area = voxel_dimensions[0] * voxel_dimensions[1]
        
        saros_pred_path = os.path.join(resampled_saros_pred_dir, image.replace('_0000.nii.gz', '.nii.gz'))
        saros_pred_nifti = nib.load(saros_pred_path)
        
        image_affine = image_nifti.affine
        image_data = image_nifti.get_fdata()
        pred_data = pred_nifti.get_fdata()
        saros_pred_data = saros_pred_nifti.get_fdata()

        # Voxel volume
        voxel_volume = np.prod(pred_nifti.header.get_zooms())

        # Set bounds
        lower_z_bound = np.median(np.where(pred_data == 26)[2]).astype(int)
        upper_z_bound = np.median(np.where(pred_data == 32)[2]).astype(int)
        
        print(f"lower_z_bound: {lower_z_bound}")
        print(f"upper_z_bound: {upper_z_bound}")

        # Create masks
        muscle_mask = (pred_data == 119)
        abd_cav_mask = (pred_data == 120)
        saros_abd_cav_mask = (saros_pred_data == 3)

        # Apply the z-bound constraints to the 3D mask
        abd_cav_mask[:, :, :lower_z_bound] = False
        abd_cav_mask[:, :, upper_z_bound + 1:] = False
        saros_abd_cav_mask[:, :, :lower_z_bound] = False
        saros_abd_cav_mask[:, :, upper_z_bound + 1:] = False

        # Extract original image voxel values in the muscle and abdomen cavity masks
        original_muscle_voxels = image_data[muscle_mask]
        original_abd_voxels = image_data[abd_cav_mask]
        original_saros_abd_voxels = image_data[saros_abd_cav_mask]

        # Calculate muscle intensity statistics
        intensity_muscle_avg = np.median(original_muscle_voxels)
        intensity_muscle_std = np.std(original_muscle_voxels)

        # SAT erosion and intensity threshold calculation
        results = erode_sat(image, image.replace('_0000.nii.gz', '.nii.gz'), resampled_image_dir, resampled_pred_dir, resampled_saros_pred_dir)
        
        #kevs_pred_nii = nib.Nifti1Image(results[17].astype(np.int16), image_nifti.affine)
        my_pred_probs = nib.Nifti1Image(results[18], image_affine)

        kevs_percentile = results[17]
        kevs_percentile[:, :, :lower_z_bound] = False
        kevs_percentile[:, :, upper_z_bound + 1:] = False
        
        kevs_percentile_vol = np.sum(kevs_percentile) * voxel_volume / 1000
        
        kevs_prob = results[18]
        kevs_prob[:, :, :lower_z_bound] = False
        kevs_prob[:, :, upper_z_bound + 1:] = False
        
        kevs_prob_vol = np.sum(kevs_prob) * voxel_volume / 1000
        
        kevs_full_range = results[19]
        kevs_full_range[:, :, :lower_z_bound] = False
        kevs_full_range[:, :, upper_z_bound + 1:] = False
        
        kevs_full_range_vol = np.sum(kevs_full_range) * voxel_volume / 1000
        
        print(f"Percentile volume: {kevs_percentile_vol}")
        print(f"Prob volume: {kevs_prob_vol}")
        print(f"full range volume: {kevs_full_range_vol}")
        
        my_pred_probs = nib.Nifti1Image(results[18], image_affine)
    
        #nib.save(my_pred_probs, os.path.join(vat_dir, image.replace('_0000.nii.gz', '_KEVS_VAT.nii.gz')))
        
        kde_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_sat_kde'
        
        kde_output_path = os.path.join(kde_dir, f'{patient_id}_{series_id}_kde.png')
        
        # Save the KDE plot for SAT and VAT masks
        save_kde_plot(
            results[7],          # kde_original_values for SAT
            results[8],          # kde_eroded_values for SAT
            results[14],        # vat_kde_data for VAT masks (comes from erode_sat)
            results[13],         # x_range
            kde_output_path,     # Output path for saving the plot
            results[3],          # original_avg
            results[4],          # original_std
            results[5],          # eroded_avg
            results[6],          # eroded_std
            results[9],          # kde_original_mean
            results[10],         # kde_original_std
            results[11],         # kde_eroded_mean
            results[12]          # kde_eroded_std
        )
        
        vat_volumes = calculate_vat_volume(original_abd_voxels, thresholds, voxel_volume)
        
        # Prepare row dictionary
        row = {
            'patient_id': patient_id,
            'series_id': series_id,
            'KEVS VAT volume': kevs_prob_vol,
            'VAT Volume 1 (-190 to -30)': vat_volumes[0],
            'VAT Volume 2 (-200 to -10)': vat_volumes[1],
            'VAT Volume 3 (-200 to -20)': vat_volumes[2],
            'VAT Volume 4 (-250 to -50)': vat_volumes[3],
            'VAT Volume 5 (-195 to -45)': vat_volumes[4]
        }
        
        for vertebrae in vertebrae_list:
            middle_slice = int(np.median(np.where(pred_data == labels[vertebrae])[2]))
            print(f"Middle {vertebrae} slice: {middle_slice}")
            vertebral_slice_mask = (kevs_prob[:, :, middle_slice] == 1)
            # Calculate the area based on the number of true values in the mask
            vat_vertebrae_area = np.sum(vertebral_slice_mask) * voxel_area /100
            
            print(f"VAT {vertebrae} area: {vat_vertebrae_area:.4f}")
            row[f"VAT_{vertebrae}_area"] = vat_vertebrae_area
            
        # Combine iliopsoas_left and iliopsoas_right, autochthon_left and autochthon_right
        iliopsoas_volume = (
            calculate_volume(image_nifti, pred_nifti, 'iliopsoas_left', labels) +
            calculate_volume(image_nifti, pred_nifti, 'iliopsoas_right', labels)
        )
        autochthon_volume = (
            calculate_volume(image_nifti, pred_nifti, 'autochthon_left', labels) +
            calculate_volume(image_nifti, pred_nifti, 'autochthon_right', labels)
        )

        row['iliopsoas_volume'] = iliopsoas_volume
        row['autochthon_volume'] = autochthon_volume

        # Iterate through the rest of the body features and vertebrae
        for body_feature in ["Muscle", "Subcutaneous_Tissue"]:
            volume_body_feature = calculate_volume(image_nifti, pred_nifti, body_feature, labels, lower_threshold=results[20], upper_threshold=results[21])
            row[f"{body_feature}_volume"] = volume_body_feature

            for vertebrae in vertebrae_list:
                body_feature_area, area_intensity_avg, area_intensity_std = calculate_vertebrae_information(image_nifti, pred_nifti, vertebrae, labels, body_feature=body_feature)
                
                # Get the information for iliopsoas_left and iliopsoas_right separately
                iliopsoas_area, iliopsoas_intensity_avg, iliopsoas_intensity_std = calculate_vertebrae_information(image_nifti, pred_nifti, vertebrae, labels, left_and_right=True, iliopsoas=True)
                
                # Get the information for autochthon_left and autochthon_right separately
                es_area, es_intensity_avg, es_intensity_std = calculate_vertebrae_information(image_nifti, pred_nifti, vertebrae, labels, left_and_right=True, iliopsoas=False)
                
                row[f"{body_feature}_{vertebrae}_intensity_avg"] = area_intensity_avg
                #row[f"{body_feature}_{vertebrae}_intensity_std"] = area_intensity_std
                row[f"{body_feature}_{vertebrae}_area"] = body_feature_area
                
                row[f"iliopsoas_{vertebrae}_area"] = iliopsoas_area
                row[f"iliopsoas_{vertebrae}_intensity_avg"] = iliopsoas_intensity_avg
                row[f"iliopsoas_{vertebrae}_intensity_std"] = iliopsoas_intensity_std
                
                row[f"autochthon_{vertebrae}_area"] = es_area
                row[f"autochthon_{vertebrae}_intensity_avg"] = es_intensity_avg
                row[f"autochthon_{vertebrae}_intensity_std"] = es_intensity_std
                
        for vertebrae in vertebrae_list:
            avg_vert_intensity, std_vert_intensity = calculate_vertebrae_intensity(image_nifti, pred_nifti, vertebrae, labels)
            row[f"{vertebrae}_intensity_avg"] = es_intensity_avg
            row[f"{vertebrae}_intensity_std"] = es_intensity_std
                

        # Append the row to the data list
        data.append(row)

    # Create a DataFrame dynamically from data and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(os.getcwd(), 'IPCAI_VAT_KEVS.csv'), index=False)

if __name__ == '__main__':
    vat_from_saros_w_vertebrae(resampled_image_dir, resampled_pred_dir, resampled_saros_pred_dir, vat_dir, labels)
    