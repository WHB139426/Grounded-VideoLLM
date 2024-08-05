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
import json
from torch.backends import cudnn
from mm_utils.utils import *


msrvtt_pred = load_json('./experiments/acc_records_msrvtt_caption_pretrain.json')
msrvtt_label = load_json('/data/hvw5451/data/msrvttqa/test_videodatainfo.json')['sentences']
msrvtt_eval_anno = {}

for item in msrvtt_pred:
    msrvtt_eval_anno[item['video_ids']] = {
        'pred': 'N/A.',
        'labels': [],
    }
for item in msrvtt_pred:
    msrvtt_eval_anno[item['video_ids']]['pred'] = item['pred_texts'].replace('.','')
for item in msrvtt_label:
    msrvtt_eval_anno[item['video_id']]['labels'].append(str(item['caption']))

msvd_pred = load_json('./experiments/acc_records_msvd_caption_pretrain.json')
msvd_label = load_json('/data/hvw5451/data/msvdqa/test_captions.json')
msvd_eval_anno = {}

for item in msvd_pred:
    msvd_eval_anno[item['video_ids']] = {
        'pred': 'N/A.',
        'labels': [],
    }
for item in msvd_pred:
    msvd_eval_anno[item['video_ids']]['pred'] = item['pred_texts'].replace('.','')
for item in msvd_label:
    msvd_eval_anno[item['video_id']]['labels'] = item['captions']

"""
eval_anno = {
    'video_id_1': {
        'pred': xxxxxxxx,
        'labels': [yyyyy, yyyyyy, yyyyy, ......]
    }
    'video_id_2': {
        'pred': xxxxxxxx,
        'labels': [yyyyy, yyyyyy, yyyyy, ......]
    }
    ...
}
"""


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu4(eval_anno,):
    all_bleu4_scores = 0
    pred_anno = {}
    gt_anno = {}
    for key in eval_anno.keys():
        pred_anno[key] = str(eval_anno[key]['pred'])
        gt_anno[key] = eval_anno[key]['labels']
    for key in pred_anno.keys():
        predicted_caption = pred_anno[key]
        ground_truth_captions = gt_anno[key]
        predicted_caption = predicted_caption.split()
        reference_captions = [ref.split() for ref in ground_truth_captions]
        # 计算 BLEU-4 分数
        smoothing_function = SmoothingFunction().method4
        all_bleu4_scores += sentence_bleu(reference_captions, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    return all_bleu4_scores/len(pred_anno)

from pycocoevalcap.cider.cider import Cider

def compute_cider(eval_anno):
    pred_anno = {}
    gt_anno = {}
    for key in eval_anno.keys():
        pred_anno[key] = [str(eval_anno[key]['pred'])]
        gt_anno[key] = eval_anno[key]['labels']
    # 计算 CIDEr 分数
    score, _ = Cider().compute_score(gt_anno, pred_anno)
    return score, _

score = compute_bleu4(msrvtt_eval_anno,)
print(f'MSRVTT BLEU-4 Score: {score}')

score, _ = compute_cider(msrvtt_eval_anno)
print(f'MSRVTT CIDEr Score: {score}', _)

score = compute_bleu4(msvd_eval_anno,)
print(f'MSVD BLEU-4 Score: {score}')

score, _ = compute_cider(msvd_eval_anno)
print(f'MSVD CIDEr Score: {score}')




