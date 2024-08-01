import sys
sys.path.append('.')
import urllib.request
import urllib.error
import pdb
import argparse
import os
import json
import requests


def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

parser = argparse.ArgumentParser()
parser.add_argument("--video_directory", type=str, default='/data3/whb/data/DiDeMo/videos/', help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", default=True)
args = parser.parse_args()

if args.download:
    assert os.path.exists(args.video_directory)

multimedia_template = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4'


caps = [] 
caps.extend(read_json('/data3/whb/data/DiDeMo/train_data.json'))
caps.extend(read_json('/data3/whb/data/DiDeMo/val_data.json'))
caps.extend(read_json('/data3/whb/data/DiDeMo/test_data.json'))
videos = set([cap['video'] for cap in caps])

def download_video(url, save_dir, filename):
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 完整的文件路径
    file_path = os.path.join(save_dir, filename)

    # 请求视频数据
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        # 以二进制写入的方式打开文件
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"视频已成功下载到 {file_path}")
    else:
        print(f"下载失败，状态码: {response.status_code}, {url}")

def read_hash(hash_file):
    lines = open(hash_file).readlines()
    yfcc100m_hash = {}
    for line_count, line in enumerate(lines):
         # sys.stdout.write('\r%d/%d' %(line_count, len(lines)))
         line = line.strip().split('\t')
         yfcc100m_hash[line[0]] = line[1]

    return yfcc100m_hash

def get_aws_link(h):
     return multimedia_template %(h[:3], h[3:6], h) 

yfcc100m_hash = read_hash('/data3/whb/data/DiDeMo/yfcc100m_hash.txt')

missing_videos = []
from tqdm import tqdm
for video_count, video in enumerate(tqdm(videos)):

    video_id = video.split('_')[1]
    link = get_aws_link(yfcc100m_hash[video_id])

    file_path = os.path.join(args.video_directory, video)
    if os.path.exists(file_path):
        continue
    else:
        download_video(link, args.video_directory, video)


# nohup python download_didemo.py > download_didemo.out 2>&1 &   2816629