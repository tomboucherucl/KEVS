import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Label names and their indices
label_names = {
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
        "esophagus": 15,
        "small_bowel": 18,
        "duodenum": 19,
        "colon": 20,
        "urinary_bladder": 21,
        "prostate": 22,
    }
# List of label indices to include for the binary dice calculation
labels_of_interest = list(label_names.values())
#print(labels_of_interest)

thresholding_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/pred_190_30'
gt_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/ALL_VAT_PREDS/Edited_VAT_resampled'

kevs_pred_dir = '/media/HDD1/tom/SSM/IPCAI_2025_images/IPCAI_nifti_resampled_preds'

data = []

for thresh_pred in tqdm(sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])):
    thresh_pred_nii = nib.load(os.path.join(thresholding_dir, thresh_pred))
    thresh_pred_numpy = thresh_pred_nii.get_fdata()
    thresh_pred_int = np.zeros_like(thresh_pred_numpy, np.int16)
    thresh_pred_int[thresh_pred_numpy>0] = 1
    
    num_thresh_pred = np.sum(thresh_pred_numpy[thresh_pred_numpy==1])
    
    kevs_pred_nii = nib.load(os.path.join(kevs_pred_dir, thresh_pred))
    kevs_pred_numpy = kevs_pred_nii.get_fdata()
    kevs_pred_int = np.zeros_like(kevs_pred_numpy, np.int16)
    kevs_pred_int[kevs_pred_numpy>0] = 1
    
    saros_from_kevs = np.zeros_like(kevs_pred_numpy, np.int16)
    saros_from_kevs[kevs_pred_numpy>117] = 1
    
    ts_from_kevs = kevs_pred_int &~ saros_from_kevs
    
    ts_from_kevs_binary = np.zeros_like(kevs_pred_numpy, np.int16)
    ts_from_kevs_binary[ts_from_kevs>0] = 1
    
    both_bin = ts_from_kevs_binary & thresh_pred_int
    
    sum_both_bin = np.sum(both_bin)
    
    """ print(num_thresh_pred)
    print(sum_both_bin) """
    
    percentage_overlap = sum_both_bin/num_thresh_pred
    
    data.append(percentage_overlap)
    print(f"Percentage overlap for {thresh_pred} is {percentage_overlap:.4f}")
    
    organ_multilabel = np.zeros_like(thresh_pred_numpy, np.int16)
    multi_label_erroneous_organs = np.zeros_like(thresh_pred_numpy, np.int16)
    
    organ_preds = np.isin(kevs_pred_numpy, labels_of_interest)
    erroneous_pred = organ_preds & thresh_pred_int
    organ_erroneous_removed = organ_preds &~ erroneous_pred
    
    organ_multilabel[organ_preds] = kevs_pred_numpy[organ_preds]
    organ_multilabel[erroneous_pred>0] = 100
    
    """ print(erroneous_pred.shape)
    print(kevs_pred_numpy.shape)
    print(multi_label_erroneous_organs.shape)
    
    multi_label_erroneous_organs[left_over_organs] = kevs_pred_numpy[left_over_organs]    
    multi_label_erroneous_organs[erroneous_pred>0] = 100   """
    
    save_dir =  '/media/HDD1/tom/SSM/IPCAI_2025_images/erroneous_organ_pred'
    os.makedirs(save_dir, exist_ok=True)
    
    nib.save(nib.Nifti1Image(organ_multilabel, affine=thresh_pred_nii.affine), os.path.join(save_dir, thresh_pred))
    
print(f"average percentage of thresh vat pred overlaps with organs: {np.mean(data)*100}")

