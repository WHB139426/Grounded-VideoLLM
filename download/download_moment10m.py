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
import os
import pandas as pd
import time
import pickle
import pandas as pd
from tqdm import tqdm
import json
import requests
import subprocess
from pytube.innertube import _default_clients

_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def load_json(path):
    with open(path) as f:
        data = json.load(f)
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
                empty_files.append(file.replace('.mp4', ''))
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    return empty_files

packed_data = load_json('/data3/whb/data/Moment-10m/GESM_data.json')
video_path = '/data3/whb/data/Moment-10m/videos/'

video_names = list(packed_data.keys())
missing_urls = []

for video_name in tqdm(video_names):
    url = f'https://www.youtube.com/watch?v={video_name}'    
    file_path = os.path.join(video_path, f"{video_name}.mp4")

    if os.path.exists(file_path):
        continue
    else:
        download_video(url=url, video_dir=video_path, video_id=f"{video_name}.mp4", missing_urls=missing_urls)
    
    # download_video(url=url, video_dir=video_path, video_id=f"{video_name}.mp4", missing_urls=missing_urls)

empty_files = find_and_delete_empty_files(video_path)

missing_urls.extend(empty_files)

save_json(missing_urls, 'missing_urls_moments-10m.json')

# nohup python download_moment10m.py > download_moment10m.out 2>&1 &           1822687








