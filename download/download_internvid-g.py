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

# from pytube.innertube import _default_clients
# _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def download_video(url, video_dir, video_id):
    max_retries = 1
    backoff_factor = 1
    for attempt in range(max_retries):
        try:
            # print(f"Downloading video {video_id}")
            video_name = f"{video_id}"
            youtube = YouTube(url, use_oauth=True, allow_oauth_cache=True)
            video = youtube.streams.filter(only_video=True, subtype='mp4', res='360p').first()
            video.download(video_dir, filename=video_name)
            # print(f"Finished downloading video {video_id}")
            return None
        except (VideoPrivate, VideoUnavailable, RegexMatchError, KeyError) as err:
            print(f"Error {type(err).__name__} for URL: {url}")
            return url
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            print(f"Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(backoff_factor * (2 ** attempt))
        except Exception as e:
            print(f"{e}")
            return url
    return url

    # video_name = f"{video_id}"
    # youtube = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    # video = youtube.streams.filter(only_video=True, subtype='mp4', res='360p').first()
    # video.download(video_dir, filename=video_name)


def process_video(item, video_path):
    video_id = item['video'].replace('.mp4','')
    url = f'https://www.youtube.com/watch?v={video_id}'  
    file_path = os.path.join(video_path, f"{video_id}.mp4")

    if os.path.exists(file_path):
        return
    else:
        download_video(url, video_path, f"{video_id}.mp4")


def main():
    packed_data = load_json('/home/haibo/data/InternVid-G/filter_train.json')
    video_path = '/home/haibo/data/InternVid-G/videos/'
    missing_urls = []
    random.shuffle(packed_data)
    packed_data = packed_data[::-1]

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(process_video, item, video_path): item for item in packed_data}
        for future in tqdm(as_completed(future_to_video), total=len(future_to_video)):
            result = future.result()
            if result:
                missing_urls.append(result)

    # for item in tqdm(packed_data):
    #     process_video(item, video_path, clip_path)
    #     break

    print(len(packed_data), len(missing_urls))

if __name__ == "__main__":
    main()

# nohup python download_panda_multi.py > download_panda_multi.out 2>&1 &    1545976