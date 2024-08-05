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


def parse_args():
    parser = argparse.ArgumentParser()
    # evaluation
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--dtype', type=torch.dtype, default=torch.bfloat16, choices=[torch.bfloat16, torch.float32])

    # model && dataset
    parser.add_argument('--dataset', type=str, default='anet_grounding', choices=['msrvtt_caption', 'msvd_caption', 'anet_caption', 'charades_sta', 'qvhighlights', 'anet_grounding'])
    parser.add_argument('--model', type=str, default='llava_next_video', choices=['llava_next_video'])
    parser.add_argument('--llm', type=str, default='llama3', choices=['llama3', 'vicuna'])
    parser.add_argument('--stage', type=str, default="grounded", choices=['pretrain', 'grounded', 'sft'])
    parser.add_argument('--max_txt_len', type=int, default=2048)
    parser.add_argument('--ckpt', type=str, default='/data/hvw5451/weights/ckpt/grounded_llava_next_video_llama3_mix_grounded_multi_modal_projector_video_projecter_language_model.pth')

    parser.add_argument('--num_temporal_tokens', type=int, default=300)
    parser.add_argument('--num_frames', type=int, default=96)
    parser.add_argument('--num_segs', type=int, default=12)
    parser.add_argument('--lora', type=bool, default=True)

    # generation
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    args = parser.parse_args()
    return args

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

@torch.inference_mode()
def eval(args, val_dataset, model):
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_bs, pin_memory=True, shuffle=False, drop_last=False, num_workers=4)
    model.eval()
    acc_records = []

    for step, data in enumerate(tqdm(val_loader)):
        samples = {
                "video_ids": data['video_ids'],
                "question_ids": data['question_ids'],
                "prompts": data['prompts'],
                "temporal_pixel_values":data['temporal_pixel_values'].to(args.device),
                "spatial_pixel_values": data['spatial_pixel_values'].to(args.device),
            }
        generate_kwargs = {
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "temperature":args.temperature,
            "top_p":args.top_p,
            }
        with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype): # 前后开启autocast
            with torch.inference_mode():
                pred_texts = model.generate(samples, **generate_kwargs)

        for i in range(len(data['video_ids'])):
            acc_records.append(
                {
                    "video_ids": data['video_ids'][i],
                    "question_ids": data['question_ids'][i],
                    "prompts": data['prompts'][i],
                    "pred_texts": pred_texts[i],
                    "answers": data['answers'][i] if 'answers' in data.keys() else 'N/A.',
                    "durations": float(data['durations'][i]) if 'durations' in data.keys() else 'N/A.',
                }
            )
        print(acc_records[-1]['prompts'])
        print(acc_records[-1]['pred_texts'])
        print(acc_records[-1]['answers'])
        # save_json(acc_records, f'./experiments/acc_records_{args.dataset}_{args.stage}.json')



if __name__ == '__main__':
    args = parse_args()
    init_seeds(args.seed)

    if args.dataset == 'msrvtt_caption':
        from datasets.msrvtt_caption import MSRVTT_Caption
        val_dataset = MSRVTT_Caption(
            video_path = "/data/hvw5451/data/msrvttqa/videos",
            anno_path = '/data/hvw5451/data/msrvttqa/test_caption.json',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )
    elif args.dataset == 'msvd_caption':
        from datasets.msvd_caption import MSVD_Caption
        val_dataset = MSVD_Caption(
            video_path = "/data/hvw5451/data/msvdqa/videos",
            anno_path = '/data/hvw5451/data/msvdqa/test_captions.json',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )
    elif args.dataset == 'anet_caption':
        from datasets.activitynet import ANet_Caption
        val_dataset = ANet_Caption(
            anno_path = "/data/hvw5451/data/activitynet/captions/val_1.json",
            video_path = '/data/hvw5451/data/activitynet/videos',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )
    elif args.dataset == 'anet_grounding':
        from datasets.activitynet import ANet_Grounding
        val_dataset = ANet_Grounding(
            anno_path = "/data/hvw5451/data/activitynet/captions/val_1.json",
            video_path = '/data/hvw5451/data/activitynet/videos',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )
    elif args.dataset == 'charades_sta':
        from datasets.charades_sta import Charades_STA
        val_dataset = Charades_STA(
            anno_path = "/data/hvw5451/data/Charades/charades_sta_test.json",
            video_path = '/data/hvw5451/data/Charades/videos',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )
    elif args.dataset == 'qvhighlights':
        from datasets.qvhighlights import QVHighlights
        val_dataset = QVHighlights(
            anno_path = "/data/hvw5451/data/qvhighlights/highlight_val_release.jsonl",
            video_path = '/data/hvw5451/data/qvhighlights/videos',
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            num_temporal_tokens = args.num_temporal_tokens,
            sample='middle',
            llm=args.llm,
        )

    if args.model == 'llava_next_video':
        from models.llava_next_video import LLAVA_NEXT_VIDEO
        model = LLAVA_NEXT_VIDEO(
            dtype=args.dtype, 
            stage=args.stage, 
            max_txt_len=args.max_txt_len, 
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            lora=args.lora,
            llm=args.llm,
            )
        if args.stage == 'pretrain':
            ckpt = torch.load(args.ckpt, map_location='cpu')['model']
            model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
            model.video_projecter.load_state_dict(ckpt['video_projecter'])
        elif args.stage == 'grounded':
            ckpt = torch.load(args.ckpt, map_location='cpu')['model']
            model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
            model.video_projecter.load_state_dict(ckpt['video_projecter'])
            model.language_model.load_state_dict(ckpt['language_model'])            

    model.to(args.device)

    print(get_parameter_number(model))
    print("val_dataset: ", len(val_dataset))
    print(args)

    eval(args, val_dataset, model)