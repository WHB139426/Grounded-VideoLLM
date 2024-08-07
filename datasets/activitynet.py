from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import re
import requests
from collections import Counter
from io import BytesIO
import json
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template

class ANet_Caption(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/activitynet/captions/train.json",
        video_path = '/data/hvw5451/data/activitynet/videos',
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
        self.prompts = []
        self.answers = []

        # save_files = []


        for key in self.data.keys():
            item = self.data[key]
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

            prompt_conv = [
                {"from": "human", "value": "<image>\n"+instruction},
                {"from": "gpt", "value": ''}                
            ]
            sep, eos = self.chat_template.separator.apply()
            prompt = self.chat_template.encode(prompt_conv).replace(eos, '')
            self.answers.append(answer)
            self.prompts.append(prompt)

        #     save_files.append(
        #         {
        #             'video_id': key,
        #             'question_id': key,
        #             'video_file': 'activitynet/videos/'+key+'.mp4',
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/data/hvw5451/data/mix_sft/activitynet-dense-caption.json')

    def __len__(self):
        return len(self.video_ids)

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

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        question_id = str(self.question_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])
        answer = str(self.answers[index])
        prompt = self.prompts[index]

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
                "answers": self.convert_time_position(answer, duration),
                "text_inputs": self.convert_time_position(text_input, duration),
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "durations":  float(duration),
            }


# dataset = ANet_Caption()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("prompts: ",             entry['prompts'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))



class ANet_Grounding(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/activitynet/captions/val_1.json",
        video_path = '/data/hvw5451/data/activitynet/videos',
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
        self.prompts = []
        self.answers = []

        for key in self.data.keys():
            item = self.data[key]
            caption_num = len(item['sentences'])
            for i in range(caption_num):
                self.question_ids.append(f"key_{i}")
                self.video_files.append(key+'.mp4')
                self.video_ids.append(key)
                caption = item['sentences'][i].strip()
                caption = caption[0].lower() + caption[1:]

                start_time = item['timestamps'][i][0]
                end_time = item['timestamps'][i][1]
                answer = f'From <{start_time}> to <{end_time}>.'

                conversations = [
                    {"from": "human", "value": "<image>\n"+f"At which time interval in the video can we see {caption[:-1]}?"},
                    {"from": "gpt", "value": answer}
                ]
                self.text_inputs.append(self.chat_template.encode(conversations))

                prompt_conv = [
                    {"from": "human", "value": "<image>\n"+f"At which time interval in the video can we see {caption[:-1]}?"},
                    {"from": "gpt", "value": ''}                
                ]
                sep, eos = self.chat_template.separator.apply()
                prompt = self.chat_template.encode(prompt_conv).replace(eos, '')
                self.answers.append(answer)
                self.prompts.append(prompt)

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
        answer = str(self.answers[index])
        prompt = self.prompts[index]

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
                "answers": self.convert_time_position(answer, duration),
                "text_inputs": self.convert_time_position(text_input, duration),
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "durations": float(duration),
            }

# dataset = ANet_Grounding()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("prompts: ",             entry['prompts'])
#     print("answers: ",             entry['answers'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("durations: ",             entry['durations'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))

def remove_time_tags(s):
    # 使用正则表达式匹配 <number> 格式并替换为空字符串
    cleaned_string = re.sub(r'^(<\d+\.\d+> )+', '', s)
    return cleaned_string.strip()

class ANet_RTL(Dataset):
    def __init__(
        self,
        anno_path = "/data/hvw5451/data/activitynet/activitynet_train_gpt-4-0613_temp_6_f10009.json",
        video_path = '/data/hvw5451/data/activitynet/videos',
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

        # save_files = []

        for key in self.data.keys():
            item = self.data[key]
            self.question_ids.append(key)
            self.video_files.append(key+'.mp4')
            self.video_ids.append(key)
            conversations = []
            for i in range(len(item['QA'])):
                conversations.append({"from": "human", "value": item['QA'][i]['q'] + ' Include the timestamps of key events.'})
                conversations.append({"from": "gpt", "value": remove_time_tags(item['QA'][i]['a'])})

            conversations[0]['value'] = "<image>\n" + conversations[0]['value']
            self.text_inputs.append(self.chat_template.encode(conversations))

        #     save_files.append(
        #         {
        #             'video_id': key,
        #             'question_id': key,
        #             'video_file': 'activitynet/videos/'+key+'.mp4',
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/data/hvw5451/data/mix_sft/ANet_RTL.json')

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
                "durations": float(duration),
            }

# dataset = ANet_RTL(llm='llama3')
# for i in range(100):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))


class VCG_Plus_112k(Dataset):
    def __init__(
        self,
        video_path = "/data/hvw5451/data/activitynet/videos",
        anno_path = '/data/hvw5451/data/activitynet/vcg-plus_112K.json',
        num_frames = 128,
        num_segs = 16,
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

        for item in self.data:
            self.question_ids.append(item['id'])
            self.video_files.append(item['id']+'.mp4')
            self.video_ids.append(item['id'])
            conversations = item['conversations']
            conversations[0]['value'] =  conversations[0]['value'].replace('<video>', '<image>')
            self.text_inputs.append(self.chat_template.encode(conversations))

        #     save_files.append(
        #         {
        #             'video_id': item['id'],
        #             'question_id': item['id'],
        #             'video_file': 'activitynet/videos/'+item['id']+'.mp4',
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/data/hvw5451/data/mix_sft/vcg_plus_112k.json')

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

# dataset = VCG_Plus_112k(llm='llama3')
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))