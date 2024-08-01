import random
import re
import requests
from PIL import Image
from io import BytesIO
import json
import os
import pickle

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

dir_names = ['coin', 'HiREST', 'Moment-10m', 'querYD', 'vitt', 'VTG-IT']

exist_videos = {}

for dir_name in dir_names:
    file_lists = os.listdir(f"/data3/whb/data/{dir_name}/videos")
    exist_videos[dir_name] = file_lists

save_json(exist_videos, 'exist_videos.json')