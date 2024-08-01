import os
import pandas as pd
from tqdm import tqdm
import pickle
import pandas as pd
from tqdm import tqdm
import json
import requests
import argparse
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
import pytube
import requests
from pytube.exceptions import RegexMatchError, VideoRegionBlocked, VideoUnavailable, VideoPrivate
from pytube.innertube import _default_clients
from pytube.exceptions import VideoPrivate, ExtractError, MembersOnly, PytubeError
import time

_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def download_video(url, video_dir, video_id, missing_urls):

    max_retries=2
    backoff_factor=1
    try:
        for attempt in range(max_retries):
            try:
                print(f"Downloading video {video_id}")
                video_name = f"{video_id}"
                youtube = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
                video = youtube.streams.first()
                video.download(video_dir, filename=video_name)
                print(f"Finished downloading video {video_id}")
                return
            except VideoPrivate as err:
                missing_urls.append(url)
                print(f"URL VideoPrivate {url}")
                pass 
            except VideoUnavailable as err:
                missing_urls.append(url)
                print(f"URL VideoUnavailable {url}")
                pass 
            except RegexMatchError as err:
                missing_urls.append(url)
                print(f"RegexMatchError {url}")
                pass 
            except KeyError as err:
                missing_urls.append(url)
                print(f"KeyError {url}")
                pass 
            except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
                print(f"Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(backoff_factor * (2 ** attempt))
            except Exception as e:
                print(f"An error occurred: {e}")
                break

    except Exception as e:
        print(f"An error occurred: {e}") 

def find_and_delete_empty_files(directory):
    empty_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                empty_files.append(file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    return empty_files

captions = read_pickle_file('/data3/whb/data/querYD/raw_captions_combined_filtered.pkl')
timestamps = read_pickle_file('/data3/whb/data/querYD/times_captions_combined_filtered.pkl')
print(captions['video-__5k7e0f3r4'])
print(timestamps['video-__5k7e0f3r4'])
        
# 读取JSON文件
packed_data = []
with open('/data3/whb/data/querYD/relevant-video-links-v2.txt', 'r') as file:
    url_list = file.readlines()
    # 去掉每行的换行符
    url_list = [url.strip() for url in url_list]

print(url_list[0][32:])
print(len(url_list))
video_path = '/data3/whb/data/querYD/videos/'

missing_urls = []

for url in tqdm(url_list):

    id = url[32:]

    file_path = os.path.join(video_path, f"{id}.mp4")

    if os.path.exists(file_path):
        continue
    else:
        download_video(url=url, video_dir=video_path, video_id=f"{id}.mp4", missing_urls=missing_urls)

print(len(url_list), len(missing_urls))

empty_files = find_and_delete_empty_files(video_path)

save_json(missing_urls, 'missing_urls_queryd.json')


# nohup python download_queryd.py > download_queryd.out 2>&1 &   3426806