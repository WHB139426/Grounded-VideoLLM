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


def filter_unexist_gesm(data, file_path='/home/haibo/data/Moment-10m/videos'):
    exist_files = os.listdir(file_path)
    fiter_files = []
    for key in tqdm(data.keys()):
        if f'{key}.mp4' in exist_files:
            item = data[key]
            captions = item['captions']
            timestamps = item['timestamps']
            if len(captions) != 0:
                fiter_files.append({
                    "video_id": key,
                    "captions": captions,
                    "timestamps": timestamps
                })
    return fiter_files


# data = load_json("/home/haibo/data/Moment-10m/GESM_data.json")
# fiter_files = filter_unexist_gesm(data)
# save_json(fiter_files, "/home/haibo/data/Moment-10m/simplified_GESM_data.json")
# print(len(data), len(fiter_files))


def filter_unexist_moment_10m(data, file_path='/home/haibo/data/Moment-10m/videos'):
    exist_files = os.listdir(file_path)
    fiter_files = []

    for key_video in tqdm(data.keys()):
        if f'{key_video}.mp4' in exist_files:
            item = data[key_video]
            for key_type in item.keys():
                sub_item = item[key_type]
                index = 0
                for subsub_item in sub_item:
                    fiter_files.append({
                        "video_id": key_video,
                        "q_id": f"{key_video}_{key_type}_{index}",
                        "data_type": key_type,
                        "variables": subsub_item["variables"],
                        "conversations": subsub_item["conversations"],
                        "clip_similarity": subsub_item["clip_similarity"] if "clip_similarity" in subsub_item.keys() else 'N/A.',
                    })
                    index += 1

    return fiter_files

# data = load_json("/home/haibo/data/Moment-10m/Moment-10M_0.json")
# data2 = load_json("/home/haibo/data/Moment-10m/Moment-10M_1.json")
# data.update(data2)
# fiter_files = filter_unexist_moment_10m(data)
# save_json(fiter_files, "/home/haibo/data/Moment-10m/simplified_Moment_10M.json")
# print(len(data), len(fiter_files))


class Moment10M_GESM(Dataset):
    def __init__(
        self,
        anno_path = "/home/haibo/data/Moment-10m/simplified_GESM_data.json",
        video_path = "/home/haibo/data/Moment-10m/videos",
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

        for item in self.data:
            self.question_ids.append(item['video_id'])
            self.video_files.append(item['video_id']+'.mp4')
            self.video_ids.append(item['video_id'])
            answer = self.convert_dense_captions(item['captions'], item['timestamps'])
            conversations = [
                {"from": "human", "value": "<image>\n"+random.choice(dense_caption_prompts)},
                {"from": "gpt", "value": answer}
            ]
            self.text_inputs.append(self.chat_template.encode(conversations))

    def __len__(self):
        return len(self.video_ids)

    def convert_dense_captions(self, captions, timestamps):
        res = []
        for cap, ts in zip(captions, timestamps):
            text = f'From <{ts[0]}> to <{ts[1]}>, {cap}'
            res.append(text)
        res = '\n'.join(res)
        return res

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
                # "text_inputs": text_input,
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }


# dataset = Moment10M_GESM()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))


















# file_names = os.listdir('/home/haibo/data/Moment-10m/videos')
# for file_name in tqdm(file_names):
#     print(file_name)
#     pixel_values, frame_indices, fps, total_frame_num = read_frames_decord(
#         video_path = os.path.join('/home/haibo/data/Moment-10m/videos', file_name),
#         num_frames = 128,
#         sample = 'rand',
#     )
#     # print(pixel_values.shape, frame_indices, fps, total_frame_num)


# dataset = Moment10M_GESM()
# from torch.utils.data import Dataset, DataLoader
# data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=16, pin_memory=True, prefetch_factor=None)
# for step, data in enumerate(tqdm(data_loader)):
#     continue

