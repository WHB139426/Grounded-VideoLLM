from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import requests
from collections import Counter
from io import BytesIO
import json
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template

def filter_unexist_data(data, file_path='/home/haibo/data/vtimellm_stage2/clips'):
    exist_files = os.listdir(file_path)
    fiter_files = []
    for item in tqdm(data):
        video_id = item['id']
        if f'{video_id}.mp4' in exist_files:
            fiter_files.append(item)
    return fiter_files

# data = load_json('/home/haibo/data/vtimellm_stage2/stage2.json')
# fiter_files = filter_unexist_data(data)
# print(len(data), len(fiter_files))
# save_json(fiter_files, "/home/haibo/data/vtimellm_stage2/simplified_train.json")

class VTimeLLM_Stage2(Dataset):
    def __init__(
        self,
        anno_path = "/home/haibo/data/vtimellm_stage2/simplified_train.json",
        video_path = '/home/haibo/data/vtimellm_stage2/clips',
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.num_temporal_tokens = num_temporal_tokens
        self.sample = sample

        self.data = load_json(anno_path)
        self.chat_template = LLaMA3_Template()

        self.video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        self.image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

        self.video_ids = []
        self.question_ids = []
        self.video_files = []
        self.text_inputs = []

        def replace_token(match):
            token = match.group(0)  # 获取匹配到的 <s0> 或 <e1> 之类的字符串
            value = tokens.get(token, token)  # 获取 token 的对应值
            return f'<{value}>'  # 用 token 的值替换原来的 <s0> 或 <e1>，并加上尖括号

        for item in self.data:
            self.question_ids.append(item['id'])
            self.video_files.append(item['id']+'.mp4')
            self.video_ids.append(item['id'])

            conversations = item['conversations']
            conversations[0]['value'] = conversations[0]['value'].replace('<video>', '<image>')
            tokens = item['meta']['token']
            # 处理 conversations 中的每个 value 字段
            for i in range(len(conversations)):
                conversations[i]['value'] = re.sub(r'<s\d+>|<e\d+>', replace_token, conversations[i]['value'])
                
            self.text_inputs.append(self.chat_template.encode(conversations))

    def __len__(self):
        return len(self.video_ids)

    def convert_time_position(self, answer, duration):
        # 定义一个函数，将匹配到的浮点数转换为整数
        def replace_float(match):
            time = float(match.group(1))
            quantized_time = int(self.num_temporal_tokens * time / duration)
            return f'<{quantized_time}>'
        # 使用正则表达式匹配所有的浮点数时间戳
        pattern = r'<(\d+\.\d+)>'
        # 替换匹配到的浮点数时间戳
        new_answer = re.sub(pattern, replace_float, answer)
        return new_answer

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        question_id = str(self.question_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])
        
        pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
            video_path = os.path.join(self.video_path, video_file),
            num_frames = self.num_frames,
            sample = self.sample,
        )

        temporal_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            temporal_pixel_values.append(self.video_processor(pixel_values[i]))
        temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)) # [num_frames, 3, 224, 224]

        num_frames_per_seg = int(self.num_frames // self.num_segs)
        indices_spatial = [(i*num_frames_per_seg) + int(num_frames_per_seg/2) for i in range(self.num_segs)]
        spatial_pixel_values = []
        for i_spatial in indices_spatial:
            spatial_pixel_values.append(self.image_processor(pixel_values[i_spatial]))
        spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)) # [num_segs, 3, 336, 336]

        return {
                "video_ids": video_id,
                "question_ids": question_id,
                "text_inputs": self.convert_time_position(text_input, duration),
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }

# dataset = VTimeLLM_Stage2()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))