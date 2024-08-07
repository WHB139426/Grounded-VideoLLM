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
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template


# dataset_names_grounding = [
#     'ANet_RTL', 'COIN', 'DiDeMo', 'HiREST', 'querYD', 'ViTT', 'VTG-IT', 'Moment-10m',
#     ]
# mix_sft_grounding = []
# for key in dataset_names_grounding:
#     data = load_json(f'/data/hvw5451/data/mix_sft/{key}.json')
#     print(key, len(data))
#     for i in range(len(data)):
#         data[i]['dataset_name'] = key
#     mix_sft_grounding += data
# print('mix_sft_grounding', len(mix_sft_grounding))
# print()
# save_json(mix_sft_grounding, '/data/hvw5451/data/mix_sft/mix_sft_grounding.json')


# dataset_names_instruction = [
#     'vcg_plus_112k', 'videochat2_conversations', 'videochat_instruct', 
#     'videochat2_egoqa', 'nextqa', 'intentqa', 'clevrer', 'webvid-qa', 'sthsthv2', 'kinetics',
#     'TextVR', 'youcook2', 'webvid-caption', 'sharegpt4video', 'msvd_caption', 'msrvtt_caption',  
#     ]
# mix_sft_instruction = []
# for key in dataset_names_instruction:
#     data = load_json(f'/data/hvw5451/data/mix_sft/{key}.json')
#     if key == 'webvid-caption':
#         data = random.sample(data, int(len(data)/10))
#     print(key, len(data))
#     for i in range(len(data)):
#         data[i]['dataset_name'] = key
#     mix_sft_instruction += data
# print('mix_sft_instruction', len(mix_sft_instruction))
# print()
# save_json(mix_sft_instruction, '/data/hvw5451/data/mix_sft/mix_sft_instruction.json')

# mix_sft = mix_sft_instruction + mix_sft_grounding
# print('mix_sft', len(mix_sft))
# save_json(mix_sft, '/data/hvw5451/data/mix_sft/mix_sft.json')

# missing = []
# video_nums = []
# data = load_json("/data/hvw5451/data/mix_sft/mix_sft.json")
# def filter_unexist(data, file_path='/data/hvw5451/data'):
#     for item in tqdm(data):
#         video_file_path = os.path.join(file_path, item['video_file'])
#         video_nums.append(video_file_path)
#         if not os.path.exists(video_file_path):
#             missing.append(item)
#             print(f"{video_file_path} not exist!!!!!!")
#     print("video_nums: ", len(list(set(video_nums))), "missing videos: ", len(list(set(missing))))
# filter_unexist(data)

class MixSFT(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/mix_sft/mix_sft.json",
        video_path = "/data/hvw5451/data",
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
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
        self.text_inputs = []
        self.dataset_names = []

        for item in self.data:
            self.question_ids.append(item['question_id'])
            self.video_files.append(item['video_file'])
            self.video_ids.append(item['video_id'])
            conversations = item['conversation']
            self.text_inputs.append(self.chat_template.encode(conversations))
            self.dataset_names.append(item['dataset_name'])

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
        dataset_name = self.dataset_names[index]
        
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
                'dataset_name': dataset_name,
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }


# dataset = MixSFT()
# for i in range(1000):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'], entry['dataset_name'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))







