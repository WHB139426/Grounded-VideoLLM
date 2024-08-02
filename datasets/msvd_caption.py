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
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template

# data = load_pkl('/home/haibo/data/msvdqa/raw-captions.pkl')
# print(data['-4wsuPCjDBc_5_15'])

# with open('/home/haibo/data/msvdqa/test_list.txt', 'r') as f:
#     reads = f.readlines() #txt中所有字符串读入data，得到的是一个list
# test_ids = [r.replace('\n','') for r in reads]

# new_data = []
# for key in data.keys():
#     if key in test_ids:
#         captions = []
#         for cap in data[key]:
#             captions.append(' '.join(cap))
#         new_data.append(
#             {
#                 'video_id': key,
#                 'captions': captions
#             }
#         )
# print(new_data[0], len(new_data))
# save_json(new_data, '/home/haibo/data/msvdqa/test_captions.json')

class MSVD_Caption(Dataset):
    def __init__(
        self,
        video_path = "/home/haibo/data/msvdqa/videos",
        anno_path = '/home/haibo/data/msvdqa/test_captions.json',
        num_frames = 128,
        num_segs = 16,
        num_temporal_tokens = 500,
        sample='rand',
        llm='llama3',
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.num_temporal_tokens = num_temporal_tokens
        self.sample = sample

        self.data = load_json(anno_path)
        if llm == 'llama3':
            self.chat_template = LLaMA3_Template()
        elif llm == 'vicuna':
            self.chat_template = Vicuna_Template()
            
        self.video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        self.image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

        self.video_ids = []
        self.question_ids = []
        self.video_files = []
        self.prompts = []
        self.answers = []

        for item in self.data:
    
            self.question_ids.append(item['video_id'])
            self.video_files.append(item['video_id']+'.avi')
            self.video_ids.append(item['video_id'])
            self.answers.append(item['captions'][0]+'.')

            conversations = [
            {"from": "human", "value": "<image>\n"+"Describe the following video concisely."},
            {"from": "gpt", "value": ''}
            ]
            sep, eos = self.chat_template.separator.apply()
            prompt = self.chat_template.encode(conversations).replace(eos, '')
            self.prompts.append(prompt)

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        question_id = str(self.question_ids[index])
        prompt = self.prompts[index]
        video_file = str(self.video_files[index])
        answer = str(self.answers[index])

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
                "prompts": prompt,
                "answers": answer,
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }


# dataset = MSVD_Caption()
# for i in range(10):
#     sample = random.choice(dataset)
#     print("video_ids: ", sample['video_ids'], "question_ids: ", sample['question_ids'])
#     print(sample['temporal_pixel_values'].shape, sample['spatial_pixel_values'].shape)
#     print("prompts: ", sample['prompts'])
#     print("answers: ", sample['answers'])
#     print()
# print(len(dataset))