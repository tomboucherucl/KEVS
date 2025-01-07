import os

vat_dir = '/media/HDD1/tom/KEVS/data_unkeyed/vat/ground_truth'
filtered_scan_dir= 'filtered_scans'
filtered_mask_dir = 'filtered_preds'

vat_mrn_list = []
for file in os.listdir(vat_dir):
    mrn = file[:8]
    vat_mrn_list.append(mrn)
    
num_removed = 0
for file in os.listdir(filtered_scan_dir):
    mrn = file[:8]
    print(mrn)
    if mrn in vat_mrn_list:
        num_removed += 1
        os.remove(os.path.join(filtered_scan_dir, file))
        os.remove(os.path.join(filtered_mask_dir, file.replace('_0000.nii.gz', '.nii.gz')))
        
print(num_removed)