import os
import numpy as np
import nibabel as nib
import pandas as pd

from scipy.stats import wilcoxon
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

thresholding_mrns = ['21207670', '21260966', '21270369', '21275957', '21276499', '21316038', '21353148','21354530','21375845','40265487']
kevs_mrns = ['21221312', '21238848', '21282741', '21419169', '40690080']
totalsegmentator_mrns = ['21272522', '21310908', '21343104', '41644655', '41688648']

ground_truth_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/Edited_VAT_resampled'
kevs_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_combined_15'
ts_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_190_30'

full_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_preds'

dice_list_kevs = []
dice_list_ts = []

dice_full_list_kevs = []
dice_full_list_ts = []

nsd_list_kevs = []
nsd_list_ts = []

prec_list_kevs = []
prec_list_ts = []

recall_list_kevs = []
recall_list_ts = []

def extract_boundaries(mask):
    """Extract the boundaries of a binary mask."""
    # Structuring element for erosion
    struct = np.ones((3, 3), dtype=bool)
    
    # Erode the mask to find the boundaries
    eroded_mask = binary_erosion(mask, structure=struct)
    
    # The boundary is the original mask minus the eroded mask
    boundary = mask & ~eroded_mask
    
    return boundary
    
def nsd_score(mask_gt, mask_pred, tolerance, voxel_spacing=(1.5, 1.5)):
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    # Extract boundaries
    boundary_gt = extract_boundaries(mask_gt)
    boundary_pred = extract_boundaries(mask_pred)

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

def sensitivity_score(truth, prediction):
    tp = np.sum((truth == 1) & (prediction == 1))
    fn = np.sum((truth == 1) & (prediction == 0))
    if tp + fn == 0:
        return 1  # Indeterminate sensitivity
    return tp / (tp + fn)

def precision_score(truth, prediction):
    tp = np.sum((truth == 1) & (prediction == 1))
    fp = np.sum((truth == 0) & (prediction == 1))
    if tp + fp == 0:
        return 1  # Indeterminate specificity
    return tp / (tp + fp)

def dice_score(pred_matrix, mask_matrix):
    if np.sum(pred_matrix)*np.sum(mask_matrix) == 0:
        return 0
    if np.sum(pred_matrix)+np.sum(mask_matrix) == 0:
        return 1
    else:
        numerator = np.sum(pred_matrix*mask_matrix)
        return (2*numerator)/(np.sum(pred_matrix) + np.sum(mask_matrix))

num_slice_list = []

for ground_truth in tqdm(sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')])):  
    print(ground_truth[:8])
    num_annotated_slices = 0
    if ground_truth[:8] in totalsegmentator_mrns:
        print(f"Patient ID: {ground_truth[:8]}")
        kevs_dc_image = []
        ts_dc_image = []      
        mask_nii = nib.load(os.path.join(ground_truth_dir, ground_truth))
        mask_numpy = mask_nii.get_fdata()
        
        print(f"The shape of this image is: {mask_numpy.shape}")
        num_axial_slices = np.array(mask_numpy.shape)[-1] 
        num_coronal_slices = np.array(mask_numpy.shape)[-2] 
        num_sag_slices = np.array(mask_numpy.shape)[-3] 
        
        kevs_pred_nii = nib.load(os.path.join(kevs_pred_dir, ground_truth))
        kevs_pred_numpy = kevs_pred_nii.get_fdata()
        
        ts_pred_nii = nib.load(os.path.join(ts_pred_dir, ground_truth))
        ts_pred_numpy = ts_pred_nii.get_fdata()
        
        full_pred_nii = nib.load(os.path.join(full_pred_dir, ground_truth))
        full_pred_numpy = full_pred_nii.get_fdata()

        lower_z_bound = np.max(np.where(full_pred_numpy == 26)[2])
        upper_z_bound = np.min(np.where(full_pred_numpy == 32)[2])
        
        """ mask_numpy[:, :, :lower_z_bound] = False
        mask_numpy[:, :, upper_z_bound + 1:] = False
        full_pred_numpy[:, :, :lower_z_bound] = False
        full_pred_numpy[:, :, upper_z_bound + 1:] = False
        kevs_pred_numpy[:, :, :lower_z_bound] = False
        kevs_pred_numpy[:, :, upper_z_bound + 1:] = False
        ts_pred_numpy[:, :, :lower_z_bound] = False
        ts_pred_numpy[:, :, upper_z_bound + 1:] = False """
        
        
        kevs_full_dice = dice_score(kevs_pred_numpy.astype(int), mask_numpy.astype(int))
        ts_full_dice = dice_score(ts_pred_numpy.astype(int), mask_numpy.astype(int))
        
        dice_full_list_kevs.append(kevs_full_dice)
        dice_full_list_ts.append(ts_full_dice)
        
        max_kevs_list = [np.max(np.where(mask_numpy==1)[2]), np.max(np.where(kevs_pred_numpy==1)[2])]
        min_kevs_list = [np.min(np.where(mask_numpy==1)[2]), np.min(np.where(kevs_pred_numpy==1)[2])]
        
        max_ts_list = [np.max(np.where(mask_numpy==1)[2]), np.max(np.where(ts_pred_numpy==1)[2])]
        min_ts_list = [np.min(np.where(mask_numpy==1)[2]), np.min(np.where(ts_pred_numpy==1)[2])]
        
        max_kevs = np.max(max_kevs_list)
        min_kevs = np.min(min_kevs_list)
        
        max_ts = np.max(max_ts_list)
        min_ts = np.min(min_ts_list)
        
        for slice in range(num_axial_slices):
            if np.sum(mask_numpy[:,:,slice]) > 0:
                num_annotated_slices += 1
                
            #if np.sum(kevs_pred_numpy[:,:,slice].astype(int)>0):
            kevs_dc_axial = dice_score(kevs_pred_numpy[:,:,slice].astype(int), mask_numpy[:,:,slice].astype(int))
            
            dice_list_kevs.append(kevs_dc_axial)
            
            kevs_dc_image.append(kevs_dc_axial)
            
            kevs_nsd = nsd_score(mask_numpy[:,:,slice], kevs_pred_numpy[:,:,slice], tolerance=2)
            nsd_list_kevs.append(kevs_nsd)
            
            kevs_precision = precision_score(mask_numpy[:,:,slice], kevs_pred_numpy[:,:,slice])
            prec_list_kevs.append(kevs_precision)
            
            kevs_recall = sensitivity_score(mask_numpy[:,:,slice], kevs_pred_numpy[:,:,slice])
            recall_list_kevs.append(kevs_recall)
                
            ts_dc_axial = dice_score(ts_pred_numpy[:,:,slice].astype(int), mask_numpy[:,:,slice].astype(int))
            
            dice_list_ts.append(ts_dc_axial)
            
            print(f"Slice: {slice}, DICE KEVS-TS: {(kevs_dc_axial-ts_dc_axial):.4f}")
            
            ts_dc_image.append(ts_dc_axial)
            
            ts_nsd = nsd_score(mask_numpy[:,:,slice], ts_pred_numpy[:,:,slice], tolerance=2)
            nsd_list_ts.append(ts_nsd)
            
            ts_precision = precision_score(mask_numpy[:,:,slice], ts_pred_numpy[:,:,slice])
            prec_list_ts.append(ts_precision)
            
            ts_recall = sensitivity_score(mask_numpy[:,:,slice], ts_pred_numpy[:,:,slice])
            recall_list_ts.append(ts_recall)
        num_slice_list.append(num_annotated_slices)
        dice_image_stat, dice_image_p_value = wilcoxon(np.array(kevs_dc_image), np.array(ts_dc_image), alternative='greater')
            
        print(dice_image_stat, dice_image_p_value)
    else:
        print(f"MRN was predicted by KEVS or TS - moving on")

print(f"Across the 20 CT scans there are: {np.sum(num_slice_list)} axial slices.")
print("-"*50)
print(f"KEVS dice: {np.mean(dice_list_kevs):.4f} +/- {np.std(dice_list_kevs):.4f}")
print(f"TS dice: {np.mean(dice_list_ts):.4f} +/- {np.std(dice_list_ts):.4f}")
dice_stat, dice_p_value = wilcoxon(np.array(dice_list_kevs), np.array(dice_list_ts), alternative='greater')
print(f'DICE Statistic: {dice_stat},  DICE p-value: {dice_p_value}')
print("-"*50)
print(f"KEVS NSD: {np.mean(nsd_list_kevs):.4f} +/- {np.std(nsd_list_kevs):.4f}")
print(f"TS NSD: {np.mean(nsd_list_ts):.4f} +/- {np.std(nsd_list_ts):.4f}")
nsd_stat, nsd_p_value = wilcoxon(np.array(nsd_list_kevs), np.array(nsd_list_ts), alternative='greater')
print(f'nsd Statistic: {nsd_stat},  nsd p-value: {nsd_p_value}')
print("-"*50)
print(f"KEVS precision: {np.mean(prec_list_kevs):.4f} +/- {np.std(prec_list_kevs):.4f}")
print(f"TS precision: {np.mean(prec_list_ts):.4f} +/- {np.std(prec_list_ts):.4f}")
prec_stat, prec_p_value = wilcoxon(np.array(prec_list_kevs), np.array(prec_list_ts), alternative='greater')
print(f'prec Statistic: {prec_stat},  prec p-value: {prec_p_value}')
print("-"*50)
print(f"KEVS recall: {np.mean(recall_list_kevs):.4f} +/- {np.std(recall_list_kevs):.4f}")
print(f"TS recall: {np.mean(recall_list_ts):.4f} +/- {np.std(recall_list_ts):.4f}")
recall_stat, recall_p_value = wilcoxon(np.array(recall_list_kevs), np.array(recall_list_ts), alternative='greater')
print(f'recall Statistic: {recall_stat},  recall p-value: {recall_p_value}')

