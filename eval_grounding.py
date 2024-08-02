import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from torch import cuda
import time
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import pickle
import random
import re
import json
from torch.backends import cudnn
from mm_utils.utils import *

def parse_time_interval(text, duration, num_temporal_tokens=300):
    # 使用正则表达式匹配所有的数字
    numbers = re.findall(r'\d+', text)
    # 将匹配到的数字转换为整数并返回
    ret = [duration*int(num)/num_temporal_tokens for num in numbers]
    if len(ret) == 2:
        return ret[0], ret[1]
    else:
        print('wrong!!')
        return 0, duration
    

def calculate_iou(pred_interval, gt_interval):
    pred_start, pred_end = pred_interval
    gt_start, gt_end = gt_interval
    
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    return intersection / union

def all_metric(data):
    # Calculate metrics
    ious = []
    recall_at_03 = 0
    recall_at_05 = 0
    recall_at_07 = 0

    for entry in data:
        duration = entry['durations']
        pred_interval = parse_time_interval(entry['pred_texts'], duration)
        gt_interval = parse_time_interval(entry['answers'], duration)

        iou = calculate_iou(pred_interval, gt_interval)
        ious.append(iou)
        
        if iou >= 0.3:
            recall_at_03 += 1
        if iou >= 0.5:
            recall_at_05 += 1
        if iou >= 0.7:
            recall_at_07 += 1

    total = len(data)
    recall_at_03 /= total
    recall_at_05 /= total
    recall_at_07 /= total
    mIoU = sum(ious) / total

    return recall_at_03, recall_at_05, recall_at_07, mIoU


charades_sta_data = load_json('./experiments/acc_records_charades_sta_grounded.json')
recall_at_03, recall_at_05, recall_at_07, mIoU = all_metric(charades_sta_data)
print(
    "Charades_STA: \n"
    "recall_@_0.3", recall_at_03, '\n', 
    "recall_@_0.5", recall_at_05, '\n',
    "recall_@_0.7", recall_at_07, '\n',
    "recall_@_mIoU", mIoU)


anet_grounding_data = load_json('acc_records_anet_grounding_grounded.json')
recall_at_03, recall_at_05, recall_at_07, mIoU = all_metric(anet_grounding_data)
print(
    "ANet_Grouding: \n"
    "recall_@_0.3", recall_at_03, '\n', 
    "recall_@_0.5", recall_at_05, '\n',
    "recall_@_0.7", recall_at_07, '\n',
    "recall_@_mIoU", mIoU)