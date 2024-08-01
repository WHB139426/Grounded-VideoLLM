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


def download_video(url, video_dir, video_id):
    max_retries = 4
    backoff_factor = 1
    video_name = f"{video_id}"
    youtube = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video = youtube.streams.first()
    video.download(video_dir, filename=video_name)

    return url

def find_and_delete_empty_files(directory):
    empty_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                empty_files.append(file)
                # os.remove(file_path)
                print(f"Deleted: {file_path}")
    return empty_files

if __name__ == "__main__":
    url = 'https://www.youtube.com/watch?v=YAEYAqQd14w'
    download_video(url, './', 'YAEYAqQd14w.mp4')
    # find_and_delete_empty_files('/data3/whb/data/panda70m_2m/clips')