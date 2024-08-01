from huggingface_hub import HfApi
from huggingface_hub import login, hf_hub_download
from huggingface_hub import snapshot_download
import zipfile
import os
from tqdm import tqdm


login()
api = HfApi()

# snapshot_download(repo_id='lmsys/vicuna-7b-v1.5', repo_type='model', local_dir='/data3/whb/weights/vicuna-7b-v1.5')
snapshot_download(repo_id='liuhaotian/llava-v1.6-vicuna-7b', repo_type='model', local_dir='/data3/whb/weights/llava-v1.6-vicuna-7b')
# hf_hub_download(repo_id='liuhaotian/llava-v1.6-vicuna-7b', filename='config.json', repo_type='model', local_dir='/data3/whb/weights/llava-v1.6-vicuna-7b')
