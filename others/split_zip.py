import os
import zipfile
from tqdm import tqdm

def zipdir(path, ziph):
    # Zip the directory contents
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))

def split_folder_into_zips(folder_path, output_dir, chunk_size=30*1024*1024*1024):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the list of files and their sizes
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_size = os.path.getsize(file_path)
            files.append((file_path, file_size))
    
    # Sort files by size (optional, can help with better distribution)
    files.sort(key=lambda x: x[1], reverse=True)
    
    # Create zip files
    current_zip = None
    current_size = 0
    zip_index = 1
    
    # Create a progress bar
    with tqdm(total=sum(file[1] for file in files), unit="B", unit_scale=True, desc="Compressing") as pbar:
        for file_path, file_size in files:
            if current_zip is None or current_size + file_size > chunk_size:
                if current_zip is not None:
                    current_zip.close()
                current_zip_path = os.path.join(output_dir, f'chunk_{zip_index}.zip')
                current_zip = zipfile.ZipFile(current_zip_path, 'w', zipfile.ZIP_DEFLATED)
                zip_index += 1
                current_size = 0

            current_zip.write(file_path, os.path.relpath(file_path, folder_path))
            current_size += file_size
            pbar.update(file_size)
    
    if current_zip is not None:
        current_zip.close()


# Run the split function
split_folder_into_zips("/home/haibo/data/Moment-10m/", "/home/haibo/data/Moment-10m/")




# nohup python split_zip.py > split_zip.out 2>&1 &      2426037

