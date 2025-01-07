import os
import nibabel as nib
import numpy as np
import torch
import shutil
from tqdm import tqdm

nifti_dir = 'niftis_new_batch_resampled'
pred_dir = 'predictions_new_batch'
sorted_nifti_list = sorted(os.listdir(nifti_dir))
sorted_pred_list = sorted(os.listdir(pred_dir))

mrn_list = []
for f in sorted_pred_list:
    if f.endswith('.nii.gz'):
        mrn = f[:8]
        mrn_list.append(mrn)
print(f"Finished extracting mrns...")
    
unique_mrn_list = np.unique(mrn_list)
print(unique_mrn_list)

chosen_list = []

for unique_mrn in unique_mrn_list:
    list_for_mrn = []
    for f in sorted_pred_list:
        if unique_mrn in f:
            list_for_mrn.append(f)
    
    print(f"The possible choices for this {unique_mrn} are {list_for_mrn}")
    
    chosen_file = None
    biggest_size = 0
    for file in list_for_mrn:
        pred_nifti = nib.load(os.path.join(pred_dir, file))
        pred_numpy = pred_nifti.get_fdata()
        if len(list(np.where(pred_numpy == 26)[2])) != 0 and len(list(np.where(pred_numpy == 32)[2])) !=0:
            tensor = torch.from_numpy(pred_numpy)
            tensor_size = tensor.numel()
            if tensor_size > biggest_size:
                biggest_size = tensor_size
                chosen_file = file
                print(f"New file chosen for {unique_mrn}: {file}, with size {tensor_size}")
    
    chosen_list.append(chosen_file)
            
scan_move_dir = 'filtered_scans'
pred_move_dir = 'filtered_preds'

os.makedirs(scan_move_dir, exist_ok=True)
os.makedirs(pred_move_dir, exist_ok=True)

for file in tqdm(chosen_list):
    scan_path = os.path.join(nifti_dir, file.replace('.nii.gz', '_0000.nii.gz'))
    pred_path = os.path.join(pred_dir, file)
    
    shutil.copy2(scan_path, os.path.join(scan_move_dir, file.replace('.nii.gz', '_0000.nii.gz')))
    shutil.copy2(pred_path, os.path.join(pred_move_dir, file))