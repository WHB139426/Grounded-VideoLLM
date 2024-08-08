import os
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    with tqdm(total=sum(file[1] for file in files), unit="B", unit_scale=True, desc=f"Compressing {os.path.basename(folder_path)}") as pbar:
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

def compress_directory(dir_name):
    input_dir = f"/home/haibo/data/{dir_name}"
    output_dir = f"/data/hvw5451/data_zip/{dir_name}"
    split_folder_into_zips(input_dir, output_dir)

# Run the split function using multiple threads
dir_names = ['activitynet', 'clevrer', 'coin', 'DiDeMo', 'HiREST', 'internvid', 'InternVid-G', 'kinetics', 
'mix_sft', 'Moment-10m', 'msrvttqa', 'msvdqa', 'nextqa', 'panda70m_2m', 'querYD', 'sharegpt4video', 'sthsthv2', 
'TextVR', 'VideoChat_instruct', 'videochat2_conversations', 'videochat2_egoqa', 'vitt', 'VTG-IT',
 'vtimellm_stage2', 'webvid-703k', 'webvid-caption', 'webvid-qa', 'youcook2']

# Using ThreadPoolExecutor to parallelize the compression tasks
with ThreadPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(compress_directory, dir_name) for dir_name in dir_names]
    for future in as_completed(futures):
        future.result()  # To raise exceptions if any occurred
