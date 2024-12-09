import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.stats import gaussian_kde, kurtosis, skew
from scipy.integrate import simpson
from tqdm import tqdm
import time

def kde_stats(kde, x_range, lower_percentile_val=0.5, upper_percentile_val=99.5):
    dx = x_range[1] - x_range[0]  # Calculate spacing
    pdf = kde(x_range)
    mean = simpson(pdf * x_range, dx=dx)
    variance = simpson(pdf * (x_range - mean)**2, dx=dx)
    
    std_dev = np.sqrt(variance)
    
    # Skewness
    skewness = simpson(pdf * ((x_range - mean) / std_dev)**3, dx=dx)
    
    # Kurtosis (excess kurtosis, where normal distribution has kurtosis of 3)
    kurtosis = simpson(pdf * ((x_range - mean) / std_dev)**4, dx=dx) - 3
    
    # Calculate cumulative density
    cdf = np.cumsum(pdf) * dx  # Cumulative sum gives the CDF
    
    # Find the corresponding x values for the given percentiles
    lower_threshold = np.interp(lower_percentile_val / 100, cdf, x_range)
    upper_threshold = np.interp(upper_percentile_val / 100, cdf, x_range)
    
    return mean, np.sqrt(variance), skewness, kurtosis, lower_threshold, upper_threshold

def binary_erode_from_exterior_2d(mask_2d, structuring_element):
    mask_copy = np.copy(mask_2d)
    filled_mask = binary_fill_holes(mask_copy)
    inner_block = filled_mask & ~ mask_copy
    
    print(f"Mask copy: {np.sum(mask_copy)}")
    print(f"filled_mask: {np.sum(filled_mask)}")
    print(f"inner_block: {np.sum(inner_block)}")
    
    
    orig_num_voxels = np.sum(mask_copy)
    num_voxels = orig_num_voxels
    
    while num_voxels > int(orig_num_voxels * 0.5):
        
        print(num_voxels)
        filled_mask = binary_erosion(filled_mask, structure=structuring_element)
        true_eroded_sat = filled_mask & ~ inner_block
        num_voxels = np.sum(true_eroded_sat)
        
    return true_eroded_sat

def erode_sat(image, mask, image_dir, mask_dir, saros_mask_dir, kevs_2_pred_dir):
    time_before = time.time()
    mask_path = os.path.join(mask_dir, mask)
    mask_nii = nib.load(mask_path)
    mask_nii_data = mask_nii.get_fdata()
    mask_affine = mask_nii.affine
    
    image_path = os.path.join(image_dir, image)
    image_nii = nib.load(image_path)
    image_nii_data = image_nii.get_fdata()
    
    saros_mask_path = os.path.join(saros_mask_dir, mask)
    saros_mask_nii = nib.load(saros_mask_path)
    saros_mask_nii_data = saros_mask_nii.get_fdata()
    
    saros_abd_cav_mask = (saros_mask_nii_data == 3)
    
    sat_mask = (mask_nii_data == 118)
    eroded_sat = np.copy(sat_mask)
    
    abd_cav_mask = (mask_nii_data == 120)
    
    
    
    #print(f"abd_cav_mask size: {np.sum(saros_abd_cav_mask)}")
    
    kevs_2_ts_pred_nii = nib.load(os.path.join(kevs_2_pred_dir, mask))
    kevs_2_ts_pred_data = kevs_2_ts_pred_nii.get_fdata()  # Get the actual data array
    kevs_2_ts_pred_binary = np.zeros_like(kevs_2_ts_pred_data, dtype=np.int16)  # Initialize with data shape
    kevs_2_ts_pred_binary[kevs_2_ts_pred_data > 0] = 1  # Apply the condition to the binary array
    
    #print(f"kevs_2_ts_pred_binary size: {np.sum(kevs_2_ts_pred_binary)}")
    
    if kevs_2_ts_pred_binary.shape == abd_cav_mask.shape:
        kevs_2_abd_cav_pred = saros_abd_cav_mask &~ kevs_2_ts_pred_binary
        kevs_2_abd_cav_pred = kevs_2_abd_cav_pred.astype(bool)
    else:
        raise ValueError("Shape mismatch between abd_cav_mask and kevs_2_ts_pred_binary")
    
    #print(f"kevs_2_abd_cav_pred size: {np.sum(kevs_2_abd_cav_pred)}")
    
    
    orig_num_voxels = np.sum(sat_mask)
    num_voxels = orig_num_voxels
    structuring_element = np.ones((3, 3))
    
    """ while num_voxels > int(orig_num_voxels * 0.2):
        eroded_sat = binary_erosion(eroded_sat, structure=structuring_element)
        num_voxels = np.sum(eroded_sat) """
    
    # Find the middle slice along the z-axis
    middle_slice = int(np.median(np.where(mask_nii_data == 29)[2]))
    
    middle_slice_l1 = int(np.median(np.where(mask_nii_data == 31)[2]))
    """ vertebral_slice_mask = (mask_data[:, :, middle_slice] == 29)
    
    # Apply the mask to the 2D slice of the image (on the middle slice)
    sat_slice_voxels = sat_mask[ :, :, middle_slice][vertebral_slice_mask]
    eroded_slice_voxels = sat_mask[ :, :, middle_slice][vertebral_slice_mask] """
    
    image_slice = image_nii_data[:, :, middle_slice]
    sat_mask_slice = sat_mask[:, :, middle_slice]
    abd_mask_slice = abd_cav_mask[:,:,middle_slice]
    abd_mask_slice_l1 = abd_cav_mask[:,:,middle_slice_l1]
    
    eroded_sat_mask_slice = binary_erode_from_exterior_2d(sat_mask_slice, structuring_element)
    
    #eroded_sat_mask_slice = eroded_sat[:, :, middle_slice]
    
    #print(np.sum(eroded_sat_mask_slice.astype(np.int16)))
    
    multilabel_mask_pre_erosion = np.zeros_like(image_slice, dtype=np.int16)
    
    multilabel_mask_pre_erosion[sat_mask_slice] = 118
    multilabel_mask_pre_erosion[abd_mask_slice] = 120
    
    multilabel_mask_pre_erosion_nii = nib.Nifti1Image(multilabel_mask_pre_erosion, image_nii.affine)
    
    multilabel_mask_post_erosion = np.zeros_like(image_slice, dtype=np.int16)
    multilabel_mask_post_erosion[eroded_sat_mask_slice > 0] = 118
    multilabel_mask_post_erosion[abd_mask_slice > 0] = 120
    
    multilabel_mask_post_erosion_nii = nib.Nifti1Image(multilabel_mask_post_erosion, image_nii.affine)
    
    eroded_sat_mask = np.zeros_like(image_slice, dtype=np.int16)

    #eroded_sat_mask[eroded_sat] = 118    
    
    eroded_sat_mask[sat_mask_slice > 0] = 118
    
    image_slice_nifti = nib.Nifti1Image(image_slice, image_nii.affine)
    eroded_sat_slice_nifti = nib.Nifti1Image(sat_mask_slice.astype(np.int16), image_nii.affine)
    
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/image_slices', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/sat_slices', exist_ok=True)
    
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pre_erosion', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/post_erosion', exist_ok=True)
    
    nib.save(image_slice_nifti, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/image_slices', image))
    nib.save(eroded_sat_slice_nifti, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/sat_original_slices', image.replace('_0000.nii.gz', '.nii.gz')))
    
    nib.save(multilabel_mask_pre_erosion_nii, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pre_erosion', image.replace('_0000.nii.gz', '.nii.gz')))
    nib.save(multilabel_mask_post_erosion_nii, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/post_erosion', image.replace('_0000.nii.gz', '.nii.gz')))
    
    # Step 4: Extract intensity values where the SAT and eroded SAT masks are True
    original_sat_intensities = image_slice[sat_mask_slice]
    eroded_sat_intensities = image_slice[eroded_sat_mask_slice]
    
    #abd_cav_mask_slice = abd_cav_mask[:, :, middle_slice]
    abd_cav_intensities = image_nii_data[abd_cav_mask]
    
    """ original_sat_intensities = image_nii_data[sat_mask]
    eroded_sat_intensities = image_nii_data[eroded_sat] """
    
    min_value = np.min(eroded_sat_intensities)
    max_value = np.max(eroded_sat_intensities)
    
    lower_bound = min_value #- np.abs(max_value - min_value)/2
    upper_bound = max_value #+ np.abs(max_value - min_value)/2
    
    x_range = np.linspace(lower_bound,
                          upper_bound, 1000)
    
    """ kde_original = gaussian_kde(original_sat_intensities)
    kde_original_values = kde_original(x_range)
    kde_original_mean, kde_original_std,  _,_,_, _ = kde_stats(kde_original, x_range) """
    
    """ kde_eroded = gaussian_kde(eroded_sat_intensities)
    kde_eroded_values = kde_eroded(x_range)
    kde_eroded_mean, kde_eroded_std, skewness, kurtosis, lower_percentile, upper_percentile = kde_stats(kde_eroded, x_range) """
    
    
    
    """ abd_cav_intensities = image_nii_data[abd_cav_mask]
    kde_abd_values = kde_eroded(abd_cav_intensities)
    lower_threshold = np.percentile(kde_abd_values, 10) 
    
    #print(f"abd cav intensities: {len(abd_cav_intensities)}")
    
    abd_kevs2_cav_intensities = image_nii_data[kevs_2_abd_cav_pred]
    
    #print(f"abd_kevs2_cav_intensities: {len(abd_kevs2_cav_intensities)}")
    kde_abd_kevs2_values = kde_eroded(abd_kevs2_cav_intensities)
    lower_kevs2_threshold = np.percentile(kde_abd_kevs2_values, 10) 
    
    abd_cav_filtered_mask = np.zeros_like(abd_cav_mask)
    abd_cav_kevs_2_filtered_mask = np.zeros_like(kevs_2_abd_cav_pred)
    
    
    abd_cav_filtered_all_range = np.zeros_like(abd_cav_mask)
    abd_cav_filtered_all_range[abd_cav_mask] = kde_abd_values 
    
    #abd_cav_filtered_mask[abd_cav_mask] = kde_abd_values < upper_threshold
    abd_cav_filtered_mask[abd_cav_mask] =  kde_abd_values > lower_threshold
    abd_cav_kevs_2_filtered_mask[kevs_2_abd_cav_pred] =  kde_abd_kevs2_values > lower_kevs2_threshold
    time_after = time.time()
    KEVS_pred_time = time_after - time_before """
    
    #print(np.sum(abd_cav_kevs_2_filtered_mask))
    
    
    
    # Apply the intensity thresholding directly on the 3D mask
    vat_masks = {
        'vat_mask_190_30': (image_nii_data >= -190) & (image_nii_data <= -30) & saros_abd_cav_mask,
        'vat_mask_200_10': (image_nii_data >= -200) & (image_nii_data <= -10) & saros_abd_cav_mask,
        'vat_mask_200_20': (image_nii_data >= -200) & (image_nii_data <= -20) & saros_abd_cav_mask,
        'vat_mask_250_50': (image_nii_data >= -250) & (image_nii_data <= -50) & saros_abd_cav_mask,
        'vat_mask_195_45': (image_nii_data >= -195) & (image_nii_data <= -45) & saros_abd_cav_mask,
        #'my_method_kde_threshold': abd_cav_filtered_mask
        #'my_method_std': (image_nii_data >= (kde_eroded_mean - 3*kde_eroded_std)) & (image_nii_data <= (kde_eroded_mean + 3*kde_eroded_std)) & abd_cav_mask,
        #'my_method_percentile': (image_nii_data >= lower_percentile) & (image_nii_data <= upper_percentile) & abd_cav_mask,
        #'KEVS_combined': abd_cav_filtered_mask,
        #'KEVS_separated': abd_cav_kevs_2_filtered_mask,
        #'my_method_full_range': abd_cav_filtered_all_range,
    }
    
    """ multilabel_mask = np.zeros_like(image_nii_data, dtype=np.int16)
    
    # Assign label 1 to vat_mask_190_30 region
    multilabel_mask[vat_masks['vat_mask_190_30']] = 3

    # Assign label 2 to my_method_percentile region
    multilabel_mask[vat_masks['my_method_probability']] = 4
    
    multilabel_nii = nib.Nifti1Image(multilabel_mask, affine=image_nii.affine) """
    #multilabel_mask_abd = np.zeros_like(image_nii_data, dtype=np.int16)
    
    #multilabel_mask_abd[abd_cav_mask] = 118
    #multilabel_mask_abd[abd_cav_mask &~ vat_masks['KEVS_combined']] = 120
    
    #multilabel_mask_abd_nii = nib.Nifti1Image(multilabel_mask_abd, image_nii.affine)
    #os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/multilabel_mask_abd', exist_ok=True)
    #nib.save(multilabel_mask_abd_nii, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/multilabel_mask_abd', image.replace('_0000.nii.gz', '.nii.gz')))
    
    #os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_combined_15', exist_ok=True)
    #percentile_pred = nib.Nifti1Image(vat_masks['my_method_percentile'].astype(np.int16), image_nii.affine)
    #kevs_combined_pred = nib.Nifti1Image(vat_masks['KEVS_combined'].astype(np.int16), image_nii.affine)
    #kevs_separated_pred = nib.Nifti1Image(vat_masks['KEVS_separated'].astype(np.int16), image_nii.affine)
    #full_range_pred = nib.Nifti1Image(vat_masks['my_method_full_range'].astype(np.int16), image_nii.affine)
    
    #os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_combined_10', exist_ok=True)
    #os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_separated_10', exist_ok=True)
    #os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/full_range', exist_ok=True)
    
    #nib.save(kevs_combined_pred, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_combined_10', image.replace('_0000.nii.gz', '.nii.gz')))
    #nib.save(kevs_separated_pred, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KEVS_separated_10', image.replace('_0000.nii.gz', '.nii.gz')))
    #nib.save(percentile_pred, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/percentile', image.replace('_0000.nii.gz', '.nii.gz')))
    #nib.save(full_range_pred, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/full_range', image.replace('_0000.nii.gz', '.nii.gz')))
    ### Thresholding preds ###
    
    pred_190_30 = nib.Nifti1Image(vat_masks['vat_mask_190_30'].astype(np.int16), image_nii.affine)
    pred_200_10 = nib.Nifti1Image(vat_masks['vat_mask_200_10'].astype(np.int16), image_nii.affine)
    pred_200_20 = nib.Nifti1Image(vat_masks['vat_mask_200_20'].astype(np.int16), image_nii.affine)
    pred_250_50 = nib.Nifti1Image(vat_masks['vat_mask_250_50'].astype(np.int16), image_nii.affine)
    pred_195_45 = nib.Nifti1Image(vat_masks['vat_mask_195_45'].astype(np.int16), image_nii.affine)
    
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_190_30', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_200_10', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_200_20', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_250_50', exist_ok=True)
    os.makedirs('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_195_45', exist_ok=True)
    
    nib.save(pred_190_30, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_190_30', image.replace('_0000.nii.gz', '.nii.gz')))
    nib.save(pred_200_10, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_200_10', image.replace('_0000.nii.gz', '.nii.gz')))
    nib.save(pred_200_20, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_200_20', image.replace('_0000.nii.gz', '.nii.gz')))
    nib.save(pred_250_50, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_250_50', image.replace('_0000.nii.gz', '.nii.gz')))
    nib.save(pred_195_45, os.path.join('/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_195_45', image.replace('_0000.nii.gz', '.nii.gz')))
    
    # Compute KDE for each VAT mask
    vat_kde_data = {}
    for mask_name, vat_mask in vat_masks.items():
        print(mask_name)
        vat_mask_slice = vat_mask[:, :, middle_slice]
        vat_intensities = image_slice[vat_mask_slice]
        
        
        if len(vat_intensities) > 0:
            #print(f"len(vat_intensities):{len(vat_intensities)}")
            kde_vat = gaussian_kde(vat_intensities)
            kde_vat_values = kde_vat(x_range)
            kde_vat_mean, kde_vat_std, _,_,_, _ = kde_stats(kde_vat, x_range)
            vat_kde_data[mask_name] = {
                'kde_values': kde_vat_values,
                'mean': kde_vat_mean,
                'std': kde_vat_std
            }
    
    """ return (sat_mask, eroded_sat, mask_affine, 
            np.mean(original_sat_intensities), np.std(original_sat_intensities),
            np.mean(eroded_sat_intensities), np.std(eroded_sat_intensities),
            kde_original_values, kde_eroded_values, kde_original_mean, kde_original_std,
            kde_eroded_mean, kde_eroded_std, x_range, vat_kde_data, kurtosis, skewness, vat_masks['KEVS_combined'].astype(np.int16), lower_threshold, upper_threshold, KEVS_pred_time) """

def save_kde_plot(kde_original_values, kde_eroded_values, vat_kde_data, x_range, output_path,
                  original_avg, original_std, eroded_avg, eroded_std,
                  kde_original_mean, kde_original_std, kde_eroded_mean, kde_eroded_std):
    plt.figure(figsize=(10, 6))
    
    # Plot the original and eroded SAT KDE
    #plt.plot(x_range, kde_original_values, label='Original SAT KDE', color='blue')
    plt.plot(x_range, kde_eroded_values)
    
    # Plot the VAT KDEs
    """ for mask_name, kde_data in vat_kde_data.items():
        plt.plot(x_range, kde_data['kde_values'], label=f'{mask_name}', linestyle='--') """
    
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.title('GKDE fit to eroded L3 SAT mask ')
    plt.legend()
    plt.grid(True)
    
    """ # Display statistics for eroded SAT KDE in the plot
    text_str = (f'Eroded SAT: Mean = {eroded_avg:.2f}, Std = {eroded_std:.2f}\n'
                f'KDE Eroded SAT: Mean = {kde_eroded_mean:.2f}, Std = {kde_eroded_std:.2f}')
    
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)) """
    
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    image_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_niftis_resampled_refined'
    mask_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_preds'
    """ eroded_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_sat_eroded'
    original_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_sat_original' """
    kde_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/KDE_display'
    saros_mask_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/baseline_saros_preds'
    kevs_sep_ts_pred = '/media/HDD1/tom/SSM/IPCAI_2025_images/baseline_ts_preds'
    # Create directories if they do not exist
    """ os.makedirs(eroded_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True) """
    edited_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/Edited_VAT_resampled'
    os.makedirs(kde_dir, exist_ok=True)
    
    data = []
    
    kevs_pred_times = []
    
    skews = []
    kurtoses = []

    for pred in tqdm(sorted([f for f in os.listdir(edited_pred_dir) if f.endswith('.nii.gz')])):
        mask = pred
        image = pred.replace('.nii.gz', '_0000.nii.gz')
        patient_id = pred[:8]
        series_id = pred[18:26]
        
        # Call the erode_sat function to get results including VAT KDE data
        results = erode_sat(image, mask, image_dir, mask_dir, saros_mask_dir, kevs_sep_ts_pred)
        
        """ kde_output_path = os.path.join(kde_dir, f'{patient_id}_{series_id}_kde.png')
        
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

        skews.append(results[16])
        kurtoses.append(results[15])
        print(f"Skewness: {results[16]}")
        print(f"kurtosis: {results[15]}")
        print(f"Running skewness average : {np.mean(skews)} +/- {np.std(skews)}")
        print(f"Running excess kurtosis average : {np.mean(kurtoses)} +/- {np.std(kurtoses)}")
        
        kevs_pred_times.append(results[-1])
        
        print(f"kevs pred time for image: {pred} is {results[-1]:.4f}")
        
        print(f"KEVS pred time running average: {np.mean(kevs_pred_times)}")

        # Prepare data for CSV
        row = {
            'patient_id': patient_id,
            'series_id': series_id,
            'SAT pred kurtosis': results[15],
            'SAT pred skewness': results[16]
        }
        
        data.append(row)
        
    df = pd.DataFrame(data)
    #df.to_csv('sat_skew_kurtosis.csv', index=False) """