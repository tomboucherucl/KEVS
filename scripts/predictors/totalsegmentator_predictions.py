import os
import subprocess
import shutil

def totalsegmentator_inference(scan_dir):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "predictions", "TotalSegmentator")
    os.makedirs(output_dir, exist_ok=True)
    scans = sorted([f for f in os.listdir(scan_dir) if f.endswith('.nii.gz')])
    for scan in scans:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        command = f"TotalSegmentator -i {scan_dir}/{scan} -o {output_dir}/{scan.removesuffix('.nii.gz')} -ta tissue_types"
        subprocess.run(command, shell=True, check=True, env=env)
        case_path = os.path.join(f"{output_dir}", scan.removesuffix('.nii.gz'))
        if os.path.isdir(case_path):
            torso_fat_file = os.path.join(case_path, "torso_fat.nii.gz")
            if os.path.exists(torso_fat_file):
                new_file_path = os.path.join(output_dir, scan)
                
                # Move and rename the file
                os.rename(torso_fat_file, new_file_path)
                
                # Delete the prediction directory
                shutil.rmtree(case_path)


    