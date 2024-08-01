import os
import pandas as pd
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import json
import requests
import time

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_csv(path):
    df = pd.read_csv(path)
    data = df.to_dict(orient='records')
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def download_video(url, save_dir, filename):
    max_retries=5
    backoff_factor=1
    try:
        # 创建保存目录（如果不存在）
        save_dir = os.path.join(save_dir, filename.split('/')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 完整的文件路径
        file_path = os.path.join(save_dir, filename.split('/')[1])
        # Retry logic
        for attempt in range(max_retries):
            try:
                # Request video data
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    # Write video data to file in binary mode
                    with open(file_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                file.write(chunk)
                    # print(f"Video successfully downloaded to {file_path}")
                    return
                else:
                    print(f"Download failed, status code: {response.status_code}")
                    break
            except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
                print(f"Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(backoff_factor * (2 ** attempt))
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        else:
            print("Failed to download video after multiple attempts.")
    except Exception as e:
        print(f"An error occurred: {e}")


packed_data = load_json('/data3/whb/data/webvid-caption/filtered_train.json')
video_path = "/data3/whb/data/webvid-caption/videos/"

download_video(packed_data[0]['contentUrl'], video_path, packed_data[0]['video'])

for item in tqdm(packed_data):
    url = item['contentUrl']
    file_path = os.path.join(video_path, item['video'])

    if os.path.exists(file_path):
        continue
    else:
        download_video(url, video_path, item['video'])


# nohup python download_webvid_caption.py > download_webvid_caption.out 2>&1 &            3445791