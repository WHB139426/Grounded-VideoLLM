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

# /home/haibo/workspace/miniconda3/envs/videollama/lib/python3.9/site-packages/pytubefix/__cache__

def download_video(url, video_dir, video_id):
    max_retries = 4
    backoff_factor = 1
    video_name = f"{video_id}"
    youtube = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video = youtube.streams.first()
    video.download(video_dir, filename=video_name)

    return url


if __name__ == "__main__":
    url = 'https://www.youtube.com/watch?v=YAEYAqQd14w'
    download_video(url, './', 'YAEYAqQd14w.mp4')
