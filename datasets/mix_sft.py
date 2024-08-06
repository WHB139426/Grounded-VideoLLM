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


dataset_names_grounding = [
    'ANet_RTL', 'COIN', 'DiDeMo', 'HiREST', 'querYD', 'ViTT', 'VTG-IT', 'Moment-10m',
    ]
mix_sft_grounding = []
for key in dataset_names_grounding:
    data = load_json(f'/data/hvw5451/data/mix_sft/{key}.json')
    print(key, len(data))
    for i in range(len(data)):
        data[i]['dataset_name'] = key
    mix_sft_grounding += data
print('mix_sft_grounding', len(mix_sft_grounding))
save_json(mix_sft_grounding, '/data/hvw5451/data/mix_sft/mix_sft_grounding.json')


dataset_names_instruction = [
    'vcg_plus_112k', 'videochat2_conversations', 'videochat_instruct', 
    'videochat2_egoqa', 'nextqa', 'intentqa', 'clevrer', 'webvid-qa', 'sthsthv2', 
    'TextVR', 'youcook2', 'webvid-caption', 'sharegpt4video', 
    ]
mix_sft_instruction = []
for key in dataset_names_instruction:
    data = load_json(f'/data/hvw5451/data/mix_sft/{key}.json')
    print(key, len(data))
    for i in range(len(data)):
        data[i]['dataset_name'] = key
    mix_sft_instruction += data
print('mix_sft_instruction', len(mix_sft_instruction))
save_json(mix_sft_instruction, '/data/hvw5451/data/mix_sft/mix_sft_instruction.json')

mix_sft = mix_sft_instruction + mix_sft_grounding
print(len(mix_sft))
save_json(mix_sft, '/data/hvw5451/data/mix_sft/mix_sft.json')
