from huggingface_hub import HfApi
from huggingface_hub import login
import zipfile
import os
from tqdm import tqdm


# scp -P 2222 -r haibo@128.173.236.231:/home/haibo/data/webvid-703k.zip /data3/whb/data/

def count_zip_files(directory):
    zip_files = [file for file in os.listdir(directory) if file.endswith('.zip')]
    return len(zip_files)

def delete_zip_files(directory):
    zip_files = [file for file in os.listdir(directory) if file.endswith('.zip')]
    for file in zip_files:
        os.remove(os.path.join(directory, file))

login()
api = HfApi()


dir_name = 'internvid'
num_file = count_zip_files(f"/data4/whb/data/{dir_name}")
print(num_file)
for i in range(num_file):
    api.upload_file(
        path_or_fileobj=f"/data4/whb/data/{dir_name}/chunk_{i+1}.zip",
        path_in_repo=f"chunk_{i+1}.zip",
        repo_id=f"WHB139426/{dir_name}",
        repo_type="dataset",
    )
delete_zip_files(f"/data3/whb/data/{dir_name}/")