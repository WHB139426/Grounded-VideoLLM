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

class NEXTQA(Dataset):
    def __init__(
        self,
        video_path = "/home/haibo/data/nextqa/videos",
        anno_path = '/home/haibo/data/nextqa/train.json',
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
            self.question_ids.append(item['video'].replace('.mp4',''))
            self.video_files.append(item['video'])
            self.video_ids.append(item['video'].replace('.mp4',''))
            conversations = []
            for i in range(len(item['QA'])):
                conversations.append({"from": "human", "value": item['QA'][i]['i'] + ' ' + item['QA'][i]['q']})
                conversations.append({"from": "gpt", "value": item['QA'][i]['a']})
            conversations[0]['value'] = "<image>\n" + conversations[0]['value']

            self.text_inputs.append(self.chat_template.encode(conversations))

        #     save_files.append(
        #         {
        #             'video_id': item['video'].replace('.mp4', ''),
        #             'question_id': item['video'].replace('.mp4', ''),
        #             'video_file': 'nextqa/videos/'+item['video'],
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/home/haibo/data/mix_sft/nextqa.json')

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

# dataset = NEXTQA()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))







# mapper = load_json('/home/haibo/data/nextqa/map_vid_vidorID.json')
# data = load_csv('/home/haibo/data/nextqa/intentqa_train.csv')
# train_intentqa = []
# for item in data:
#     video = mapper[str(item['video_id'])]
#     question = item['question']
#     question = question[0].upper() + question[1:]
#     a0 = item['a0']
#     a1 = item['a1']
#     a2 = item['a2']
#     a3 = item['a3']
#     a4 = item['a4']
#     answers_text = item[f"a{str(item['answer'])}"]
#     anwer_option = ["A", "B", "C", "D", "E"][item['answer']]
#     QA = [{"i": "", 
#         "q": f"Question: {question}?\nOptions:\n(A) {a0}.\n(B) {a1}.\n(C) {a2}.\n(D) {a3}.\n(E) {a4}.", 
#         "a": f"Answer: ({anwer_option}) {answers_text}."}]
#     train_intentqa.append({
#         'video': video+'.mp4',
#         'QA': QA
#     })
# save_json(train_intentqa, '/home/haibo/data/nextqa/train_intentqa.json')


class IntentQA(Dataset):
    def __init__(
        self,
        video_path = "/home/haibo/data/nextqa/videos",
        anno_path = '/home/haibo/data/nextqa/train_intentqa.json',
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
            self.question_ids.append(item['video'].replace('.mp4',''))
            self.video_files.append(item['video'])
            self.video_ids.append(item['video'].replace('.mp4',''))
            conversations = []
            for i in range(len(item['QA'])):
                if item['QA'][i]['q'] == '':
                    conversations.append({"from": "human", "value": item['QA'][i]['i']})
                elif item['QA'][i]['i'] == '':
                    conversations.append({"from": "human", "value": item['QA'][i]['q']})
                else:
                    conversations.append({"from": "human", "value": item['QA'][i]['i'] + ' ' + item['QA'][i]['q']})
                conversations.append({"from": "gpt", "value": item['QA'][i]['a']})
            conversations[0]['value'] = "<image>\n" + conversations[0]['value']

            self.text_inputs.append(self.chat_template.encode(conversations))

        #     save_files.append(
        #         {
        #             'video_id': item['video'].replace('.mp4', ''),
        #             'question_id': item['video'].replace('.mp4', ''),
        #             'video_file': 'nextqa/videos/'+item['video'],
        #             'conversation': conversations
        #         }
        #     )
        # save_json(save_files, '/home/haibo/data/mix_sft/intentqa.json')

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

# dataset = IntentQA()
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))
