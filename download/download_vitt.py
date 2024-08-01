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


# 读取JSON文件
packed_data = []
with open('/data3/whb/data/vitt/ViTT-annotations.json', 'r') as file:
    lines = file.readlines()
for line in lines:
    packed_data.append(json.loads(line))

print(packed_data[0])
print(packed_data[0]['annotations'][0]['timestamp'])

id_list = []
missing_urls = []
video_path = '/data3/whb/data/vitt/videos/'
for item in tqdm(packed_data):
    id = item['id']
    id_list.append(id)

    file_path = os.path.join(video_path, f"{id}.mp4")

    if os.path.exists(file_path):
        continue
    else:
        youtube_id = requests.get(f"https://data.yt8m.org/2/j/i/{id[:2]}/{id}.js", verify=False).text[10:-3]
        url = f'https://www.youtube.com/watch?v={youtube_id}'
        download_video(url=url, video_dir=video_path, video_id=f"{id}.mp4", missing_urls=missing_urls)

find_and_delete_empty_files(video_path)

save_json(missing_urls, 'missing_urls_vitt.json')
print(len(id_list), len(list(set(id_list))), len(missing_urls))

# nohup python download_vitt.py > download_vitt.out 2>&1 &                     3862122