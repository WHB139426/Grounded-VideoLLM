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



def parse_time_interval(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def process_video(item, video_path, clip_path):
    url = item['url']
    file_path = os.path.join(video_path, f"{item['video_id']}.mp4")
    clip_save_path = os.path.join(clip_path, f"{item['video_id']}.mp4")

    if os.path.exists(clip_save_path):
        return
    else:
        download_video(url, video_path, f"{item['video_id']}.mp4")

    if os.path.exists(file_path):
        try:
            moviepy_video = VideoFileClip(file_path)
            duration = moviepy_video.duration

            timestamps = eval(item['timestamp'])
            compare_list = [parse_time_interval(ts[1]) - parse_time_interval(ts[0]) for ts in timestamps]

            index = compare_list.index(max(compare_list))
            timestamp = timestamps[index]

            start_time = max(0, parse_time_interval(timestamp[0]))
            end_time = min(duration, parse_time_interval(timestamp[1]))

            clip = moviepy_video.subclip(start_time, end_time)
            clip.write_videofile(clip_save_path, audio=False)
        except Exception as e:
            print(f"Error cutting clip: {e}")

        os.remove(file_path)

    return None

def main():
    packed_data = load_json('/data3/whb/data/panda70m_2m/panda70m_training_2m.json')
    video_path = '/data3/whb/data/panda70m_2m/videos/'
    clip_path = '/data3/whb/data/panda70m_2m/clips/'
    missing_urls = []
    random.shuffle(packed_data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(process_video, item, video_path, clip_path): item for item in packed_data}
        for future in tqdm(as_completed(future_to_video), total=len(future_to_video)):
            result = future.result()
            if result:
                missing_urls.append(result)

    # for item in tqdm(packed_data):
    #     process_video(item, video_path, clip_path)
    #     break


    save_json(missing_urls, 'missing_urls_panda.json')
    print(len(packed_data), len(missing_urls))

if __name__ == "__main__":
    main()

# nohup python download_panda_multi.py > download_panda_multi.out 2>&1 &    1545976