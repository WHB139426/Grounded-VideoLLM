import random
import re
import requests
from PIL import Image
from io import BytesIO
import json
import os
import pickle
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop, ToPILImage
from typing import Optional, Tuple, Any, Union, List


dense_caption_prompts_detail = [
    "Detect and list each event in detail and its corresponding timestamps that appears in the video.",
    "Determine the start and end times of all activity events in detail, accompanied by descriptions.",
    "Capture and describe the activity events in detail, specifying their respective time intervals.",
    "Identify, timestamp, and describe various activity events occurring in the video without omission. The timestamp should include the start time and end time in seconds.",
    "Examine the video and enumerate all events you can see in detail, together with their start and end times.",
    "Perform a thorough analysis of the video and list out every event in detail with its timestamps.",
    "In the provided video, pinpoint and list all the events in detail, together with their respective time intervals.",
    "Could you outline all the events in detail and their timestamps that are visible within the video?",
]

dense_caption_prompts_short = [
    "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences.",
    "Detect and report the start and end timestamps of activity events in the video, along with descriptions.",
    "Pinpoint the time intervals of activity events in the video, and provide descriptions for each event.",
    "Can you compile a list of the activities and their timestamps featured in the video?",
    "I need you to scrutinize the video and catalog every event it contains, along with the timestamps.",
]

short_caption_prompts = [
    "Describe the following video concisely.", 
    "Provide a brief description of the given video clip.", 
    "Offer a succinct explanation of the footage presented.", 
    "Summarize the visual content of the following video.", 
    "Give a short and clear explanation of the subsequent video clip.", 
    "Share a concise interpretation of the video provided.", 
    "Present a compact description of the clip's key features.", 
    "Relay a brief, clear account of the video shown.", 
    "Render a clear and concise summary of the video below.", 
    "Write a terse but informative summary of the following video clip.", 
    "Create a compact narrative representing the video presented.",
]

vtg_prompts = [
    "When does '%s' happen in the video?",
    "At what time does the occurrence of '%s' take place in the video?",
    "During which part of the video does '%s' occur?",
    "At what point in the video does the event '%s' happen?",
    "When in the video does the '%s' incident occur?",
    "At which moment does '%s' take place in the video?",
    "During which phase of the video does '%s' happen?",
    "When does the '%s' event occur in the video?",
    "At what time does '%s' occur in the video sequence?",
    "When does the '%s' situation take place in the video?",
    "At which time interval in the video can we see '%s'?",
]

vtu_prompts = [
    "What is happening from <start> to <end>?",
    "What is taking place between <start> and <end>?",
    "What events unfold between <start> and <end>?",
    "What is happening during the period from <start> to <end>?",
    "What occurs between <start> and <end>?",
    "What is going on from <start> to <end>?",
    "How do things progress from <start> to <end>?",
    "Can you describe what happens from <start> to <end>?",
    "Describe the events occurring between <start> and <end>.",
    "Narrate the actions that unfold from <start> to <end>.",
    "Summarize the happenings between <start> and <end>.",
    "Identify the main activities occurring from <start> to <end>.",
    "Provide an overview of what happens from <start> to <end>.",
]

def _convert_to_rgb(image):
    return image.convert('RGB')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

INTERNVIDEO_MEAN = (0.485, 0.456, 0.406)
INTERNVIDEO_STD = (0.229, 0.224, 0.225)

def frame_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        ToPILImage(),
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)


def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def expand2square(pil_img, background_color=tuple(int(x*255) for x in OPENAI_DATASET_MEAN)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def load_image(image_file, pad=False):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    if pad:
        image = expand2square(image)
    return image

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)
        
def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 


