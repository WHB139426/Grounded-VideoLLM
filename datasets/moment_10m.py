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


def filter_unexist_gesm(data, file_path='/data/hvw5451/data/Moment-10m/videos'):
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


# data = load_json("/data/hvw5451/data/Moment-10m/GESM_data.json")
# fiter_files = filter_unexist_gesm(data)
# save_json(fiter_files, "/data/hvw5451/data/Moment-10m/simplified_GESM_data.json")
# print(len(data), len(fiter_files))


def filter_unexist_moment_10m(data, file_path='/data/hvw5451/data/Moment-10m/videos'):
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

# data = load_json("/data/hvw5451/data/Moment-10m/Moment-10M_0.json")
# data2 = load_json("/data/hvw5451/data/Moment-10m/Moment-10M_1.json")
# data.update(data2)
# fiter_files = filter_unexist_moment_10m(data)
# save_json(fiter_files, "/data/hvw5451/data/Moment-10m/simplified_Moment_10M.json")
# print(len(data), len(fiter_files))


class Moment10M_GESM(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/Moment-10m/simplified_GESM_data.json",
        video_path = "/data/hvw5451/data/Moment-10m/videos",
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
        llm='llama3'
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

        for item in self.data:
            self.question_ids.append(item['video_id'])
            self.video_files.append(item['video_id']+'.mp4')
            self.video_ids.append(item['video_id'])
            answer = self.convert_dense_captions(item['captions'], item['timestamps'])
            if len(item['captions']) >= 10:
                instruction = random.choice(dense_caption_prompts_detail)
            else:
                instruction = random.choice(dense_caption_prompts_short)
            conversations = [
                {"from": "human", "value": "<image>\n"+instruction},
                {"from": "gpt", "value": answer}
            ]
            self.text_inputs.append(self.chat_template.encode(conversations))

    def __len__(self):
        return len(self.video_ids)

    def convert_dense_captions(self, captions, timestamps):
        res = []
        for cap, ts in zip(captions, timestamps):
            cap = cap[0].lower() + cap[1:]
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








# file_names = os.listdir('/data/hvw5451/data/Moment-10m/videos')
# for file_name in tqdm(file_names):
#     print(file_name)
#     pixel_values, frame_indices, fps, total_frame_num = read_frames_decord(
#         video_path = os.path.join('/data/hvw5451/data/Moment-10m/videos', file_name),
#         num_frames = 128,
#         sample = 'rand',
#     )
#     # print(pixel_values.shape, frame_indices, fps, total_frame_num)


# dataset = Moment10M_GESM()
# from torch.utils.data import Dataset, DataLoader
# data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=16, pin_memory=True, prefetch_factor=None)
# for step, data in enumerate(tqdm(data_loader)):
#     continue







# data = load_json('/data/hvw5451/data/Moment-10m/simplified_Moment_10M.json')

# thresholds = {
#     'segment_caption_data': 0.145,
#     'instance_segment_caption_data': 0.146,
#     'instance_caption_data': 0.126,
#     'appearance_data': 0.129,
#     'qa_data': 0.177,
#     'instance_qa_data': 0.159,
#     'cross_segment_qa_data': 0.211,
#     'hypo_scene_data': 0.888,
#     'comp_ret_data': 0.195,
#     'segment_locate_data': 1.067
# }

# data_types = []
# for item in tqdm(data):
#     data_type = item['data_type']
#     if data_type not in data_types:
#         data_types.append(data_type)

# video_ids = []
# for item in tqdm(data):
#     video_id = item['video_id']
#     video_ids.append(video_id)
# video_ids = list(set(video_ids))

# statics_data = {}
# for video_id in tqdm(video_ids):
#     statics_data[video_id] = {}
#     for data_type in data_types:
#         statics_data[video_id][data_type] = []

# for item in tqdm(data):
#     data_type = item['data_type']
#     video_id = item['video_id']
#     statics_data[video_id][data_type].append(item)

# new_data = []
# for video_id in tqdm(statics_data.keys()):
#     item = statics_data[video_id]
#     for d_type in item.keys():
#         data_list = item[d_type]
#         for i in range(len(data_list)):
#             if data_list[i]['clip_similarity'] == 'N/A.':
#                 data_list[i]['clip_similarity'] = 0
#             data_list[i]['clip_similarity'] = float(data_list[i]['clip_similarity'])

#         data_list.sort(key=lambda x: x['clip_similarity'], reverse=True)

#         if len(data_list) > 0:
#             if data_list[0]['clip_similarity'] > thresholds[d_type]:
#                 new_append = data_list[0]
#                 new_data.append(new_append)

# print(len(new_data))
# save_json(new_data, '/data/hvw5451/data/Moment-10m/less_simplified_Moment_10m.json')





class Moment_10m(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/Moment-10m/less_simplified_Moment_10m.json",
        video_path = "/data/hvw5451/data/Moment-10m/videos",
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
        llm='llama3'
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

        for item in self.data:
            self.question_ids.append(item['q_id'])
            self.video_files.append(item['video_id']+'.mp4')
            self.video_ids.append(item['video_id'])

            variables = list(item['variables'].keys())

            conversations = []
            for dialogue in item['conversations']:
                conversations.append({"from": "human", "value": dialogue['User']},)
                conversations.append({"from": "gpt", "value": dialogue['Assistant']},)
            conversations[0]['value'] = "<image>\n" + conversations[0]['value']
            text_input = self.chat_template.encode(conversations)

            for var in variables:
                if var == 'moment':
                    timestamps = item['variables'][var]
                    if not isinstance(timestamps[0], list):
                        rep = f'<{timestamps[0]}> to <{timestamps[1]}>'
                    else:
                        rep = ''
                        for ts in timestamps:
                            rep += f'<{ts[0]}> to <{ts[1]}> and '
                        rep = rep[:-5]
                elif var == 'SOURCE_CLIP':
                    timestamps = item['variables'][var]
                    rep = f'the video from <{timestamps[0]}> to <{timestamps[1]}>'
                elif var == 'content':
                    rep = item['variables'][var]
                elif var == 'instance_class':
                    rep = item['variables'][var]
                elif var == 'click_position':
                    timepoint = float(item['variables'][var][0])
                    if '{'+'click_position'+'}' in text_input:
                        position = text_input.find('{'+'click_position'+'}')
                        if text_input[position-2:position-1] == '>':
                            rep = f'At the point of time <{timepoint}>, '
                        else:    
                            rep = f'the point of time <{timepoint}>'

                text_input = text_input.replace('{'+f'{var}'+'}', rep)

                for i in range(len(conversations)):
                    conversations[i]['value'] = conversations[i]['value'].replace('{'+f'{var}'+'}', rep)

            self.text_inputs.append(text_input)

        #     save_files.append(
        #         {
        #             'video_id': item['video_id'],
        #             'question_id': item['q_id'],
        #             'video_file': 'Moment-10m/videos/'+item['video_id']+'.mp4',
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/data/hvw5451/data/mix_sft/Moment-10m.json')


    def __len__(self):
        return len(self.question_ids)

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
                "durations": float(duration),
            }

# dataset = Moment_10m(llm='llama3')
# for i in range(100):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'], entry['durations'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))