import os
import nibabel as nib
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import SimpleITK as sitk
import torch

logging.basicConfig(filename='extract_scan_information_otherindex.log', level=logging.INFO, format='%(asctime)s %(message)s')

image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "all_scans", "scans")
pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "all_scans", "predictions", "umamba_predictions")
vat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "all_scans", "predictions", "kevs_vat_predictions", "pd_15")


data = []

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
        "Thoracic_Cavity":121,
    }

body_features_list = ["iliopsoas_left", "iliopsoas_right", "autochthon_left", "autochthon_right", "Muscle", "Subcutaneous_Tissue", "VAT"]
vertebrae_list = ["vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1"]

def calculate_volume_information(image_nii, mask_nii, upper_z_bound, lower_z_bound, body_feature, labels):
    image_nii_data = image_nii.get_fdata() 
    mask_nii_data = mask_nii.get_fdata() 
    
    voxel_dimensions = image_nii.header.get_zooms()
    voxel_volume = np.prod(voxel_dimensions)
    
    clipped_image = image_nii_data[:, :, lower_z_bound : upper_z_bound+1]
    clipped_mask = mask_nii_data[:, :, lower_z_bound : upper_z_bound+1]
    
    if body_feature == 'VAT':
        body_feature_mask_clipped = (clipped_mask == 1)
    else:
        body_feature_mask_clipped = (clipped_mask == labels[body_feature])
    body_feature_num_voxels = np.sum(body_feature_mask_clipped)
    total_volume = body_feature_num_voxels * voxel_volume
    
    body_feature_voxels = clipped_image[body_feature_mask_clipped]
    body_feature_voxels_avg_intensity = np.mean(body_feature_voxels)
    body_feature_voxels_std_intensity = np.std(body_feature_voxels)
    
    return total_volume, body_feature_voxels_avg_intensity, body_feature_voxels_std_intensity

def calculate_vertebrae_information(image_nii, mask_nii, vat_nii, vertebral_level, body_feature, labels):
    image_data = image_nii.get_fdata() 
    mask_data = mask_nii.get_fdata() 
    vat_data = vat_nii.get_fdata()
    
    voxel_dimensions = mask_nii.header.get_zooms() 
    voxel_area = voxel_dimensions[0] * voxel_dimensions[1]
    
    
    vertebral_level_indices = np.where(mask_data == labels[vertebral_level])
    
    if np.sum(vertebral_level_indices) != 0:
        # Find the middle slice along the z-axis
        middle_slice = int(np.median(vertebral_level_indices[2])) 
        
        # Get the mask for the body feature in the middle slice
        if body_feature == 'VAT':
            vertebral_slice_mask = (vat_data[:, :, middle_slice] == 1)
        else:
            vertebral_slice_mask = (mask_data[:, :, middle_slice] == labels[body_feature])
        
        # Apply the mask to the 2D slice of the image (on the middle slice)
        image_voxels = image_data[ :, :, middle_slice][vertebral_slice_mask]
        
        if len(image_voxels) > 0:
            avg_intensity = np.mean(image_voxels)
            std_intensity = np.std(image_voxels)
        else:
            avg_intensity = 0
            std_intensity = 0
        
        # Calculate the area based on the number of true values in the mask
        body_feature_area = np.sum(vertebral_slice_mask) * voxel_area
        logging.info(f"np.sum(vertebral_slice_mask): {np.sum(vertebral_slice_mask)}")
    else:
        avg_intensity = 0
        body_feature_area = 0
        
    return body_feature_area, avg_intensity, std_intensity
    
num = 0
for image in tqdm(sorted(f for f in os.listdir(image_dir) if f.endswith('.nii.gz')), total=len(os.listdir(image_dir))):
    num += 1
    key_name = image[:-12]
    logging.info(f" Key name: {key_name}")
    logging.info("-"*50)
    
    # Set path to image-mask pair
    image_path = os.path.join(image_dir, image)
    pred_path = os.path.join(pred_dir, image.replace('_0000.nii.gz', '.nii.gz'))
    
    # Load image and mask using Nibabel
    image_nii = nib.load(image_path)
    pred_nii = nib.load(pred_path)
    vat_nii = nib.load(os.path.join(vat_dir, image.replace('_0000.nii.gz', '.nii.gz')))
    
    # Extract data as numpy array
    image_nii_data = image_nii.get_fdata()
    pred_nii_data = pred_nii.get_fdata()
    
    # Calculate affine information for image
    image_nii_affine = image_nii.affine
    
    # Calculate voxel informaiton
    voxel_dimensions = image_nii.header.get_zooms() 
    voxel_volume = np.prod(voxel_dimensions)
    
    lower_z_bound = np.min(np.where(pred_nii_data == labels["vertebrae_L5"])[2])
    upper_z_bound = np.max(np.where(pred_nii_data == labels["vertebrae_L1"])[2])
    
    row = {
        'key_name': key_name,
    }   
    
    # Iterate through each tissue in the body features list
    for body_feature in body_features_list:    
        if body_feature == 'VAT':
            volume_body_feature, volume_intensity_avg, volume_intensity_std = calculate_volume_information(image_nii, vat_nii, upper_z_bound, lower_z_bound, body_feature, labels)
            logging.info(f"VOLUMETRIC: Volume (mm^3) {body_feature}: {volume_body_feature}, Avg intensity: {volume_intensity_avg}")
        else: 
            volume_body_feature, volume_intensity_avg, volume_intensity_std = calculate_volume_information(image_nii, pred_nii, upper_z_bound, lower_z_bound, body_feature, labels)
            logging.info(f"VOLUMETRIC: Volume (mm^3) {body_feature}: {volume_body_feature}, Avg intensity: {volume_intensity_avg}")
        
        row[f"{body_feature}_volume"] = volume_body_feature
        row[f"{body_feature}_intensity_avg"] = volume_intensity_avg
        row[f"{body_feature}_intensity_std"] = volume_intensity_std
        
        # Iterate through vertebrae and calculate intensity/area at specific vertebral levels
        for vertebrae in vertebrae_list:
            if body_feature == 'VAT':
                body_feature_area, area_intensity_avg, area_intensity_std = calculate_vertebrae_information(image_nii, pred_nii, vat_nii, vertebrae, body_feature, labels)
                logging.info(f"AREA: Body feature: {body_feature}, vertebra: {vertebrae}, area (mm^2): {body_feature_area}, avg intensity: {area_intensity_avg}")
            else:
                body_feature_area, area_intensity_avg, area_intensity_std = calculate_vertebrae_information(image_nii, pred_nii, vat_nii, vertebrae, body_feature, labels)
                logging.info(f"AREA: Body feature: {body_feature}, vertebra: {vertebrae}, area (mm^2): {body_feature_area}, avg intensity: {area_intensity_avg}")
            
            # Add these values to the row, keyed by the tissue and vertebrae
            row[f"{body_feature}_{vertebrae}_intensity_avg"] = area_intensity_avg
            row[f"{body_feature}_{vertebrae}_intensity_std"] = area_intensity_std
            row[f"{body_feature}_{vertebrae}_area"] = body_feature_area
        # Append the row to the data list
    data.append(row)
    
# Convert the data list into a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('extracted_image_data.csv', index=False)

        
    

        
    
        
    
        
            
    
    
    
    
    
    
    