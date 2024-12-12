import os
import shutil

thresholding_mrns = ['21207670', '21260966', '21270369', '21275957', '21276499', '21316038', '21353148','21354530','21375845','40265487']
kevs_mrns = ['21221312', '21238848', '21282741', '21419169', '40690080']
totalsegmentator_mrns = ['21272522', '21310908', '21343104', '41644655', '41688648']

combined_file_names = thresholding_mrns + kevs_mrns + totalsegmentator_mrns

file_key = {filename: f"UCLH-Cyst_{i+1:04}" for i, filename in enumerate(combined_file_names)}
    
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
    
