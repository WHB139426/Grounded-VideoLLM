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


class MixGrounded(Dataset):
    def __init__(
        self,
        moment_anno_path = "/home/haibo/data/Moment-10m/simplified_GESM_data.json",
        moment_video_path = "/home/haibo/data/Moment-10m/videos",
        vtimellm_anno_path = "/home/haibo/data/vtimellm_stage2/simplified_train.json",
        vtimellm_video_path = '/home/haibo/data/vtimellm_stage2/clips',
        anet_anno_path = "/home/haibo/data/activitynet/captions/train.json",
        anet_video_path = '/home/haibo/data/activitynet/videos',
        internvidg_anno_path = "/home/haibo/data/InternVid-G/simplified_filter_train.json",
        internvidg_video_path = '/home/haibo/data/InternVid-G/videos',
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
        llm='llama3',
    ):
        self.moment_video_path = moment_video_path
        self.vtimellm_video_path = vtimellm_video_path
        self.anet_video_path = anet_video_path
        self.internvidg_video_path = internvidg_video_path
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.num_temporal_tokens = num_temporal_tokens
        self.sample = sample

        self.moment_data = load_json(moment_anno_path)
        self.vtimellm_data = load_json(vtimellm_anno_path)
        self.anet_data = load_json(anet_anno_path)
        self.internvidg_data = load_json(internvidg_anno_path)

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

        def replace_token(match):
            token = match.group(0)  # 获取匹配到的 <s0> 或 <e1> 之类的字符串
            value = tokens.get(token, token)  # 获取 token 的对应值
            return f'<{value}>'  # 用 token 的值替换原来的 <s0> 或 <e1>，并加上尖括号

        for item in self.moment_data:
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
            self.dataset_names.append('moment-10m')

        for item in self.vtimellm_data:
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
            self.dataset_names.append('vtimellm-stage2')

        for key in self.anet_data.keys():
            item = self.anet_data[key]
            self.question_ids.append(key)
            self.video_files.append(key+'.mp4')
            self.video_ids.append(key)
            answer = self.convert_dense_captions(item['sentences'], item['timestamps'])
            if len(item['sentences']) >= 10:
                instruction = random.choice(dense_caption_prompts_detail)
            else:
                instruction = random.choice(dense_caption_prompts_short)
            conversations = [
                {"from": "human", "value": "<image>\n"+instruction},
                {"from": "gpt", "value": answer}
            ]
            self.text_inputs.append(self.chat_template.encode(conversations))
            self.dataset_names.append('anet-caption')

        for item in self.internvidg_data:
            video_id = item['video'].replace('.mp4','')
            self.question_ids.append(video_id)
            self.video_files.append(video_id+'.mp4')
            self.video_ids.append(video_id)
            caption = item['caption']
            start_time = item['start_sec']
            end_time = item['end_sec']

            random_number = random.random()
            if random_number < 0.3:
                answer = caption[0].upper() + caption[1:] + '.'
                instruction = random.choice(vtu_prompts).replace('<start>', f'<{start_time}>').replace('<end>', f'<{end_time}>')
                conversations = [
                    {"from": "human", "value": "<image>\n"+instruction},
                    {"from": "gpt", "value": answer}
                ]
            else:
                answer = f'From <{start_time}> to <{end_time}>.'
                instruction = random.choice(vtg_prompts).replace("'%s'", caption)
                conversations = [
                    {"from": "human", "value": "<image>\n"+instruction},
                    {"from": "gpt", "value": answer}
                ]

            self.text_inputs.append(self.chat_template.encode(conversations))
            self.dataset_names.append('internvid-g')

    def convert_dense_captions(self, captions, timestamps):
        res = []
        for cap, ts in zip(captions, timestamps):
            cap = cap.strip()
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

    def pos_text_coefficient(self, frame_indices, total_frame_num):
        time_pos = [self.num_temporal_tokens*i/total_frame_num for i in frame_indices]
        time_pos_left = ''
        time_pos_right = ''
        coefficient_left = []
        coefficient_right = []
        for t in time_pos:
            t_left = int(np.floor(t))
            t_right = int(np.ceil(t))
            time_pos_left += f'<{t_left}>'
            time_pos_right+= f'<{t_right}>'
            if t == t_left:
                coefficient_left.append(0.5)
                coefficient_right.append(0.5)
            else:
                coefficient_left.append(t_right-t)
                coefficient_right.append(t-t_left)

        return time_pos_left, time_pos_right, torch.tensor(coefficient_left), torch.tensor(coefficient_right)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        question_id = str(self.question_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])
        dataset_name = self.dataset_names[index]

        if dataset_name == 'moment-10m':
            video_path = os.path.join(self.moment_video_path, video_file)
        elif dataset_name == 'vtimellm-stage2':
            video_path = os.path.join(self.vtimellm_video_path, video_file)
        elif dataset_name == 'anet-caption':
            video_path = os.path.join(self.anet_video_path, video_file)
        elif dataset_name == 'internvid-g':
            video_path = os.path.join(self.internvidg_video_path, video_file)

        pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
            video_path = video_path,
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

        # time_pos_left, time_pos_right, coefficient_left, coefficient_right = self.pos_text_coefficient(frame_indices, total_frame_num)

        return {
                "video_ids": video_id,
                "question_ids": question_id,
                "text_inputs": self.convert_time_position(text_input, duration),
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "dataset_names": dataset_name,

                # "time_pos_left": time_pos_left,
                # "time_pos_right": time_pos_right,
                # "coefficient_left": coefficient_left,
                # "coefficient_right": coefficient_right,
            }

# dataset = MixGrounded(llm='llama3')
# for i in range(100):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'], entry['dataset_names'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     # print("time_pos_left: ",             entry['time_pos_left'])
#     # print(entry['time_pos_left'])
#     # print("time_pos_right: ",             entry['time_pos_right'])
#     # print("coefficient_left: ",             entry['coefficient_left'].shape)
#     # print("coefficient_right: ",             entry['coefficient_right'].shape)
#     print()
# print(len(dataset))






# llm = 'vicuna'

# from transformers import AutoTokenizer, AutoConfig
# if llm == 'llama3':
#     tokenizer = AutoTokenizer.from_pretrained("/data3/whb/weights/Meta-Llama-3-8B-Instruct", use_fast=False, truncation_side="left")
#     tokenizer.pad_token_id = 128001
# elif llm == 'vicuna':
#     tokenizer = AutoTokenizer.from_pretrained("/data3/whb/weights/vicuna-7b-v1.5", use_fast=False, truncation_side="left")

# special_token_list = [f'<{i}>' for i in range(300 + 1)]
# tokenizer.add_tokens(special_token_list)

# prompts = ['<0><2><288><300>', '<4><100><250><260>', 'From <41> to <49>, A colorful and diverse array of clothing is displayed on a rack, showcasing various patterns and designs.', 'pull up the hair to reserve place for the hair extensions from <33> to <36>.']
# prompt_tokens = tokenizer(
#     prompts,
#     return_tensors="pt",
#     padding="longest",
#     truncation=True,
# ).input_ids[:, 1:]
# print(prompt_tokens)

# label_list = []
# for tokens in prompt_tokens:
#     temp_list = []
#     for i in tokens:
#         temp_list.append(tokenizer.decode([i], skip_special_tokens=False))
#     label_list.append(temp_list)

# for t in label_list:
#     print(t)
# print(tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False))