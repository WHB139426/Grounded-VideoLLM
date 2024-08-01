import os
import pandas as pd
from tqdm import tqdm


import pandas as pd
from tqdm import tqdm
import json
import requests

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
    # 创建保存目录（如果不存在）
    save_dir = os.path.join(save_dir, filename.split('/')[0])
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 完整的文件路径
    file_path = os.path.join(save_dir, filename.split('/')[1])
    # 请求视频数据
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # 以二进制写入的方式打开文件
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"视频已成功下载到 {file_path}")
    else:
        print(f"下载失败，状态码: {response.status_code}")

packed_data = load_json('/data3/whb/data/webvid-703k/filtered_chat.json')
video_path = "/data3/whb/data/webvid-703k/videos/"

def remove_duplicates(data):
    seen_ids = set()
    unique_data = []
    
    for item in data:
        video_id = item['video']
        if video_id not in seen_ids:
            seen_ids.add(video_id)
            unique_data.append(item)
    
    return unique_data

packed_data = remove_duplicates(packed_data)


for item in tqdm(packed_data):
    url = item['contentUrl']
    file_path = os.path.join(video_path, item['video'])

    if os.path.exists(file_path):
        continue
    else:
        download_video(url, video_path, item['video'])

# nohup python download_webvid_703k.py > download_webvid_703k.out 2>&1 &             3464309
