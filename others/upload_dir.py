from huggingface_hub import HfApi
from huggingface_hub import login
import zipfile
import os
from tqdm import tqdm



login()
api = HfApi()


# api.upload_folder(
#     folder_path=f"/data3/whb/code/FSDP",
#     path_in_repo=f"./",
#     repo_id=f"WHB139426/Grounded-VideoLLM",
#     repo_type="model",
# )

# api.upload_folder(
#     folder_path=f"/data3/whb/code/download",
#     path_in_repo=f"./",
#     repo_id=f"WHB139426/Grounded-VideoLLM",
#     repo_type="model",
# )

# api.upload_folder(
#     folder_path=f"/data3/whb/weights",
#     path_in_repo=f"weights",
#     repo_id=f"WHB139426/weights",
#     repo_type="model",
# )

dir_names = ['activitynet', 'clevrer', 'coin', 'DiDeMo', 'HiREST', 'internvid', 'InternVid-G', 'kinetics', 
'mix_sft', 'Moment-10m', 'msrvttqa', 'msvdqa', 'nextqa', 'panda70m_2m', 'querYD', 'sharegpt4video', 'sthsthv2', 
'TextVR', 'VideoChat_instruct', 'videochat2_conversations', 'videochat2_egoqa', 'vitt', 'VTG-IT',
 'vtimellm_stage2', 'webvid-703k', 'webvid-caption', 'webvid-qa', 'youcook2']

api.upload_folder(
    folder_path=f"/data/hvw5451/data_zip/activitynet",
    path_in_repo=f"./",
    repo_id=f"WHB139426/Grounded-VideoLLM",
    repo_type="dataset",
)