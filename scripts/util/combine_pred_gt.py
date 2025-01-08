import os
import nibabel as nib
import numpy as np




path_gt = os.path.join('/media/HDD1/tom/KEVS/data/vat/ground_truth/UCLH-Cyst_0019.nii.gz')
path_pred = os.path.join('/media/HDD1/tom/KEVS/data/KEVS_VAT/pd_15/UCLH-Cyst_0019.nii.gz')

gt_nii = nib.load(path_gt)
gt_numpy = gt_nii.get_fdata().astype(np.int16)

pred_nii = nib.load(path_pred)
pred_numpy = pred_nii.get_fdata().astype(np.int16)

multi_label_mask = np.zeros_like(gt_numpy, np.int16)

under_pred = gt_numpy &~ pred_numpy
over_pred = pred_numpy &~ gt_numpy

#print(f"size of wrong pred for : {pred_dir}, {gt} is {np.sum(erroneous_pred)}")

multi_label_mask[under_pred>0] = 1
multi_label_mask[over_pred>0] = 2

nib.save(nib.Nifti1Image(multi_label_mask, affine=gt_nii.affine), os.path.join(os.getcwd(), 'combined_0019.nii.gz'))
    
        