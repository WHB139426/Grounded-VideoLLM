import random
import re
import requests
from PIL import Image
from io import BytesIO
import json
import os
import pickle
from tqdm import tqdm
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop, ToPILImage
from typing import Optional, Tuple, Any, Union, List

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

exist_videos = load_json('./others/exist_videos.json')

for key in exist_videos.keys():
    video_path = f"/data/hvw5451/data/{key}/videos"
    should_exist_files = exist_videos[key]
    current_files = os.listdir(video_path)
    print(key, len(should_exist_files), len(current_files))

    # Delete files in current_files that are not in should_exist_files
    for file in tqdm(current_files):
        if file not in should_exist_files:
            file_path = os.path.join(video_path, file)
            os.remove(file_path)
            print(f"Deleted {file_path}")