from huggingface_hub import HfApi
from huggingface_hub import login
import zipfile
import os
from tqdm import tqdm



login()
api = HfApi()


api.upload_folder(
    folder_path=f"/data3/whb/code/FSDP",
    path_in_repo=f"./",
    repo_id=f"WHB139426/Grounded-VideoLLM",
    repo_type="model",
)

api.upload_folder(
    folder_path=f"/data3/whb/code/download",
    path_in_repo=f"./",
    repo_id=f"WHB139426/Grounded-VideoLLM",
    repo_type="model",
)

api.upload_folder(
    folder_path=f"/data3/whb/weights",
    path_in_repo=f"weights",
    repo_id=f"WHB139426/weights",
    repo_type="model",
)
