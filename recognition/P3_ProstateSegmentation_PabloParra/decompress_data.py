# decompress_data.py
# --------------------
# Utility script for decompressing all .nii.gz archives in the HipMRI dataset.
# This will unpack every .nii.gz file inside the given root directory and save
# the resulting .nii files in the same folders. Existing .nii files will not be overwritten.

import gzip
import shutil
import os
import glob

def decompress_all(root_dir):
    """
    Recursively find all .nii.gz files inside root_dir and decompress them
    into .nii files in the same folder.
    """
    gz_files = glob.glob(os.path.join(root_dir, '**', '*.nii.gz'), recursive=True)
    print(f"Found {len(gz_files)} compressed files.")

    for gz_file in gz_files:
        nii_out = gz_file.replace('.nii.gz', '.nii')
        if os.path.exists(nii_out):
            print(f"Skipping existing file: {nii_out}")
            continue
        try:
            with gzip.open(gz_file, 'rb') as f_in, open(nii_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed: {os.path.basename(nii_out)}")
        except Exception as e:
            print(f"Error decompressing {gz_file}: {e}")

if __name__ == '__main__':
    root = r"C:\\Users\\User\\Desktop\\Final project comp3710\\keras_slices_data"
    decompress_all(root)
    print("Decompression complete.")