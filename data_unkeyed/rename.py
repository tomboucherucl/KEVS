import os

nifti_dir = 'filtered_preds'

for file in os.listdir(nifti_dir):
    first_part = file[:26]
    #print(f"{first_part}_0000.nii.gz")
    os.rename(os.path.join(nifti_dir, file), os.path.join(nifti_dir, f"{first_part}.nii.gz"))