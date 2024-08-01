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
import os
import pandas as pd
from tqdm import tqdm
import pickle
import pandas as pd
from tqdm import tqdm
import json
import requests
import subprocess
from pytube.innertube import _default_clients
import time

_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def download_video(url, video_dir, video_id, missing_urls):


    print(f"Downloading video {video_id}")
    video_name = f"{video_id}"
    youtube = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video = youtube.streams.first()
    video.download(video_dir, filename=video_name)
    print(f"Finished downloading video {video_id}")

    # max_retries=2
    # backoff_factor=1
    # try:
    #     for attempt in range(max_retries):
    #         try:
    #             print(f"Downloading video {video_id}")
    #             video_name = f"{video_id}"
    #             youtube = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
    #             video = youtube.streams.first()
    #             video.download(video_dir, filename=video_name)
    #             print(f"Finished downloading video {video_id}")
    #             return
    #         except VideoPrivate as err:
    #             missing_urls.append(url)
    #             print(f"URL VideoPrivate {url}")
    #             pass 
    #         except VideoUnavailable as err:
    #             missing_urls.append(url)
    #             print(f"URL VideoUnavailable {url}")
    #             pass 
    #         except RegexMatchError as err:
    #             missing_urls.append(url)
    #             print(f"RegexMatchError {url}")
    #             pass 
    #         except KeyError as err:
    #             missing_urls.append(url)
    #             print(f"KeyError {url}")
    #             pass 
    #         except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
    #             print(f"Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...")
    #             time.sleep(backoff_factor * (2 ** attempt))
    #         except Exception as e:
    #             print(f"An error occurred: {e}")
    #             break

    # except Exception as e:
    #     print(f"An error occurred: {e}") 





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

video_path = '/data3/whb/data/coin/videos/'
packed_data = load_json('/data3/whb/data/coin/COIN.json')['database']

missing_urls = []

for key in tqdm(packed_data.keys()):
    item = packed_data[key]
    youtube_id = key
    url = f'https://www.youtube.com/watch?v={youtube_id}'
    start_time = item['start']
    end_time = item['end']
    
    file_path = os.path.join(video_path, f"{youtube_id}.mp4")

    if os.path.exists(file_path):
        continue
    else:
        download_video(url=url, video_dir=video_path, video_id=f"{youtube_id}.mp4", missing_urls=missing_urls)


find_and_delete_empty_files(video_path)

# nohup python download_coin.py > download_coin.out 2>&1 &             2588689


# nohup bash wget_download.sh > wget_download.out 2>&1 &              104943