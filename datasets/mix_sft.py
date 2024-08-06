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


dataset_names = [
    'ANet_RTL', 'COIN', 'DiDeMo', 'HiREST', 'querYD', 'ViTT', 'VTG-IT', 'Moment-10m',
    ]

mix_sft_grounding = []

for key in dataset_names:
    data = load_json(f'/data/hvw5451/data/mix_sft/{key}.json')
    print(key, len(data))
    for i in range(len(data)):
        data[i]['dataset_name'] = key
    mix_sft_grounding += data
print('mix_sft_grounding', len(mix_sft_grounding))
save_json(mix_sft_grounding, '/data/hvw5451/data/mix_sft/mix_sft_grounding.json')
