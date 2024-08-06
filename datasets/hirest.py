from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import torch
import requests
from collections import Counter
from io import BytesIO
import json
import re
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template

# def extract_quoted_content(s):
#     # 使用正则表达式查找单引号内的内容
#     match = re.search(r"'(.*?)'", s)
#     if match:
#         return match.group(1)
#     else:
#         return None

# def extract_numbers(s):
#     # 使用正则表达式查找所有浮点数
#     numbers = re.findall(r'\d+\.\d+', s)
#     # 将找到的数字转换为浮点数并存储在列表中
#     return [float(num) for num in numbers]

# def filter_unexist_grounding(data, file_path='/data/hvw5451/data/HiREST/videos'):
#     exist_files = os.listdir(file_path)
#     fiter_files = []
#     for item in data:
#         video_id = item['video'].split('/')[-1].replace('.mp4','')
#         caption = extract_quoted_content(item['QA'][0]['q'])
#         timestamp = extract_numbers(item['QA'][0]['a'])
#         if f'{video_id}.mp4' in exist_files:
#             fiter_files.append({
#                 "video_id": video_id,
#                 "caption": caption,
#                 "timestamp": timestamp
#             })
#     return fiter_files

# def remove_timestamps_and_extract(filename):
#     # 使用正则表达式提取时间戳部分
#     match = re.search(r'_(\d+)_(\d+)(?=\.\w+$)', filename)
#     if match:
#         # 提取时间戳
#         start_time = int(match.group(1))
#         end_time = int(match.group(2))
#         # 去除时间戳部分
#         new_filename = re.sub(r'_\d+_\d+(?=\.\w+$)', '', filename)
#         return new_filename, start_time, end_time
#     else:
#         return filename, None, None

# def extract_times_and_captions(s):
#     # 使用正则表达式提取所有时间戳和对应的描述
#     matches = re.findall(r'(\d+\.\d+) - (\d+\.\d+) seconds, ([^\.]+)', s)
#     times = [[float(start), float(end)] for start, end, _ in matches]
#     captions = [caption.strip() for _, _, caption in matches]
#     return times, captions

# def filter_unexist_step(data, file_path='/data/hvw5451/data/HiREST/videos'):
#     exist_files = os.listdir(file_path)
#     fiter_files = []
#     for item in data:
#         video_id, start, end = remove_timestamps_and_extract(item['video'].split('/')[-1])
#         video_id = video_id.replace('.mp4','')
#         timestamps, captions = extract_times_and_captions(item['QA'][0]['a'])
#         for i in range(len(timestamps)):
#             timestamps[i][0] += start
#             timestamps[i][1] += start
#         if f'{video_id}.mp4' in exist_files:
#             fiter_files.append({
#                 "video_id": video_id,
#                 "captions": captions,
#                 "timestamps": timestamps,
#                 "clip_timestamps": [start, end]
#             })
#     return fiter_files

# data_grounding = load_json('/data/hvw5451/data/HiREST/data_temporal_video_grounding_hirest_instruct_tvg_0.5k_hirest.json')
# filter_data_grounding = filter_unexist_grounding(data_grounding)
# print(len(data_grounding), len(filter_data_grounding))
# print(filter_data_grounding[0])

# data_step = load_json('/data/hvw5451/data/HiREST/timeit_hirest_step_instruct_action_0.5k_hirest.json')
# filter_data_step = filter_unexist_step(data_step)
# print(len(data_step), len(filter_data_step))
# print(filter_data_step[0])

# filter_data = {
#     'grounding': filter_data_grounding,
#     'step': filter_data_step
# }

# save_json(filter_data, '/data/hvw5451/data/HiREST/train.json')

class HiREST(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/HiREST/train.json",
        video_path = '/data/hvw5451/data/HiREST/videos',
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

        save_files = []

        for item in self.data['grounding']:
            self.question_ids.append(item['video_id'])
            self.video_files.append(item['video_id']+'.mp4')
            self.video_ids.append(item['video_id'])
            caption = item['caption']
            start_time = item['timestamp'][0]
            end_time = item['timestamp'][1]
            answer = f'From <{start_time}> to <{end_time}>.'
            instruction = random.choice(sft_vtg_two_end_prompts).replace("<query_placeholder>", f"'{caption}'")

            conversations = [
                {"from": "human", "value": "<image>\n"+instruction},
                {"from": "gpt", "value": answer}
            ]
            self.text_inputs.append(self.chat_template.encode(conversations))

            # save_files.append(
            #     {
            #         'video_id': item['video_id'],
            #         'question_id': item['video_id'],
            #         'video_file': 'HiREST/videos/'+item['video_id']+'.mp4',
            #         'conversation': conversations
            #     }
            # )

        for item in self.data['step']:
            self.question_ids.append(item['video_id'])
            self.video_files.append(item['video_id']+'.mp4')
            self.video_ids.append(item['video_id'])
            clip_timestamps = item['clip_timestamps']
            start = float(clip_timestamps[0])
            end = float(clip_timestamps[1])
            instruction = random.choice(sft_specific_step_prompts).replace("<start>", f"<{start}>").replace("<end>", f"<{end}>")
            answer = self.convert_dense_captions(item['captions'], item['timestamps'])
            conversations = [
                {"from": "human", "value": "<image>\n"+instruction},
                {"from": "gpt", "value": answer}
            ]
            self.text_inputs.append(self.chat_template.encode(conversations))

        #     save_files.append(
        #         {
        #             'video_id': item['video_id'],
        #             'question_id': item['video_id'],
        #             'video_file': 'HiREST/videos/'+item['video_id']+'.mp4',
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/data/hvw5451/data/mix_sft/HiREST.json')

    def __len__(self):
        return len(self.video_ids)

    def convert_dense_captions(self, captions, timestamps):
        res = []
        for cap, ts in zip(captions, timestamps):
            cap = cap.strip()
            cap = cap[0].lower() + cap[1:]
            if cap[-1] != '.':
                cap += '.'
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
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "durations":  float(duration),
            }

# dataset = HiREST()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("durations: ",             entry['durations'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))