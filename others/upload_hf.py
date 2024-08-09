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

dir_names = ['activitynet', 'clevrer', 'coin', 'DiDeMo', 'HiREST', 'internvid', 'InternVid-G', 'kinetics', 
'Moment-10m', 'msrvttqa', 'msvdqa', 'nextqa', 'panda70m_2m', 'querYD', 'sharegpt4video', 'sthsthv2', 
'TextVR', 'VideoChat_instruct', 'videochat2_conversations', 'videochat2_egoqa', 'vitt', 'VTG-IT',
 'vtimellm_stage2', 'webvid-703k', 'webvid-caption', 'webvid-qa', 'youcook2']
 
dir_names = [
    # "msvdqa", "msrvttqa", "videochat2_egoqa", "VideoChat_instruct", "videochat2_conversations", "nextqa", "TextVR",
    # "clevrer", "kinetics", "querYD", "HiREST",  "youcook2", "coin", "activitynet", "vitt", "sthsthv2", "DiDeMo", "InternVid-G", "panda70m_2m", "vtimellm_stage2", "VTG-IT", "webvid-703k", "webvid-qa", 
    "webvid-caption", "sharegpt4video", "Moment-10m"
     
]
for dir_name in dir_names:
    num_file = count_zip_files(f"/data/hvw5451/data_zip/{dir_name}")
    print(dir_name, num_file)
    for i in range(num_file):
        api.upload_file(
            path_or_fileobj=f"/data/hvw5451/data_zip/{dir_name}/chunk_{i+1}.zip",
            path_in_repo=f"./{dir_name}/chunk_{i+1}.zip",
            repo_id=f"WHB139426/Grounded-VideoLLM",
            repo_type="dataset",
        )

