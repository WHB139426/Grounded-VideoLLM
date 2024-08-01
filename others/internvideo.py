import random
import re
import requests
import json
import os
import pickle
from tqdm import tqdm
import os
import time
import json
import requests
from tqdm import tqdm
from pathlib import Path
from pytubefix import YouTube
from pytubefix.exceptions import RegexMatchError, VideoRegionBlocked, VideoUnavailable, VideoPrivate
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


def parse_time_interval(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def remove_duplicates(data_list):
    seen_ids = set()
    unique_list = []
    
    for item in tqdm(data_list):
        youtube_id = item.get('YoutubeID')
        if youtube_id not in seen_ids:
            unique_list.append(item)
            seen_ids.add(youtube_id)
    return unique_list

def filter_top_half_by_score(data_list):
    # 按 UMT_Score 降序排序
    sorted_list = sorted(data_list, key=lambda x: x['UMT_Score'], reverse=True)
    # 只保留前1/4元素
    half_length = len(sorted_list) // 4
    return sorted_list[:half_length]

data = load_jsonl('/data3/whb/data/internvid/InternVid-10M-flt.jsonl')
print(len(data))
data = filter_top_half_by_score(data)

new_data = []
for item in tqdm(data):
    s_t = parse_time_interval(item['Start_timestamp'])
    e_t = parse_time_interval(item['End_timestamp'])
    duration = e_t-s_t
    if duration >= 10 and duration <= 40:
        new_data.append(item)
print(len(new_data))

new_data = remove_duplicates(new_data)
print(len(new_data))

save_json(new_data, '/data3/whb/data/internvid/InternVid-10M-flt-filter.json')