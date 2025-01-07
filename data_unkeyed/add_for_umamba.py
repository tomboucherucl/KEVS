import os

sorted_list = sorted(os.listdir('niftis_new_batch'))

for f in sorted_list:
    new_name = f.replace('.nii.gz', '_0000.nii.gz')
    print(new_name)
    os.rename(f'niftis_new_batch/{f}', f'niftis_new_batch/{new_name}')
    print(f"Renamed niftis_new_batch/{f} to niftis_new_batch/{new_name}")