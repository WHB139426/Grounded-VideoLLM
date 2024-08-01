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
from mm_utils.video_utils import read_frames_decord
from datasets.chat.base_template import LLaMA3_Template

def remove_duplicates(data):
    seen_ids = set()
    unique_list = []
    
    for item in data:
        if item['video'] not in seen_ids:
            unique_list.append(item)
            seen_ids.add(item['video'])
    
    return unique_list

# for i in range(frames.shape[0]):
#     frame = frames[i]
#     # 转换为 (316, 600, 3) 的形状
#     frame = frame.permute(1, 2, 0).numpy()
#     image = Image.fromarray(frame)
#     image.save(f'/data3/whb/code/FSDP/frames/frame_{i + 1}.jpg')

class Webvid_232k(Dataset):
    def __init__(
        self,
        anno_path = "/home/haibo/data/webvid-703k/filtered_train.json",
        video_path = "/home/haibo/data/webvid-703k/videos",
        num_frames = 128,
        num_segs = 16,
        num_temporal_tokens = 500,
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
            self.question_ids.append(item['id'])
            self.video_files.append(item['video'])
            self.video_ids.append(item['video'])

            conversations = item['conversations']
            conversations[0]['value'] = conversations[0]['value'].replace('\n<video>', '\n<image>')
            self.text_inputs.append(self.chat_template.encode(conversations))

    def __len__(self):
        return len(self.video_ids)

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
                "text_inputs": text_input,
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }



# dataset = Webvid_232k()
# for i in range(1000):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))


# dataset = Webvid_232k()
# from torch.utils.data import Dataset, DataLoader
# data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=16, pin_memory=True, prefetch_factor=None)
# for step, data in enumerate(tqdm(data_loader)):
#     continue
