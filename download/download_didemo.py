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
parser.add_argument("--video_directory", type=str, default='/home/haibo/data/DiDeMo/videos/', help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", default=True)
args = parser.parse_args()

if args.download:
    assert os.path.exists(args.video_directory)

multimedia_template = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4'


caps = [] 
caps.extend(read_json('/home/haibo/data/DiDeMo/train_data.json'))
caps.extend(read_json('/home/haibo/data/DiDeMo/val_data.json'))
caps.extend(read_json('/home/haibo/data/DiDeMo/test_data.json'))
videos = set([cap['video'] for cap in caps])


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

yfcc100m_hash = read_hash('/home/haibo/data/DiDeMo/yfcc100m_hash.txt')
from tqdm import tqdm
for video_count, video in enumerate(tqdm(videos)):

    video_id = video.split('_')[1]
    link = get_aws_link(yfcc100m_hash[video_id])
    video = video.split('.')[0]

    try:
        response = urllib.request.urlopen(link)
        urllib.request.urlretrieve(response.geturl(), '{}/{}.mp4'.format(args.video_directory, video))
    except:
        print("Could not download link: {}\n".format(link))
    

# nohup python download_didemo.py > download_didemo.out 2>&1 &   2816629