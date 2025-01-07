import os
import shutil

nifti_dir = 'filtered_scans'
pred_dir = 'filtered_preds'

all_mrns = []
for file in sorted(os.listdir(nifti_dir)):
    mrn = file[:8]
    all_mrns.append(mrn)

file_key = {filename: f"UCLH-Cyst_{i+21:04}" for i, filename in enumerate(sorted(all_mrns))}

with open("key_notext.txt", "a") as f:
    for file in sorted(os.listdir(nifti_dir)):
        key_name = file_key[file[:8]]
        print(f"{file[:8]}, {key_name}", file=f)
    

for file in sorted(os.listdir(nifti_dir)):
    key_name = file_key[file[:8]]
    new_scan_path = os.path.join(nifti_dir, file).replace(file, key_name+'_0000.nii.gz')
    new_scan_path = new_scan_path.replace("filtered_scans", "filtered_scans_relabelled")
    shutil.copy2(os.path.join(nifti_dir, file), new_scan_path)
    
    new_pred_path = os.path.join(pred_dir, file.replace('_0000.nii.gz', '.nii.gz')).replace(file.replace('_0000.nii.gz', '.nii.gz'), key_name+'.nii.gz')
    new_pred_path = new_pred_path.replace("filtered_preds", "filtered_preds_relabelled")
    shutil.copy2(os.path.join(pred_dir, file.replace('_0000.nii.gz', '.nii.gz')), new_pred_path)
    
print(f"all copied")
    
""" for root, dir, files in os.walk('/media/HDD1/tom/KEVS/data', topdown=True):
    for file in files:
        if file.endswith(".nii.gz"):
            key_name = file_key[file[:8]]
            original_path = os.path.abspath(os.path.join(root, file))
            if "scans" in original_path:
                new_path = original_path.replace(file, key_name+'_0000.nii.gz')
                new_path = new_path.replace("data", "keyed_data")
                os.makedirs(new_path[:-27], exist_ok=True)
                shutil.copy2(original_path, new_path)
            else:
                new_path = original_path.replace(file, key_name+'.nii.gz')
                new_path = new_path.replace("data", "keyed_data")
                print(new_path)
                print(new_path[:-22])
                os.makedirs(new_path[:-22], exist_ok=True)
                shutil.copy2(original_path, new_path) """
    
