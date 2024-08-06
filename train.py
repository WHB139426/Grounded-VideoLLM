import os
import random
from typing import Callable, Optional
import numpy as np
import torch
import draccus
import torch.distributed as dist
import argparse
from training.fsdp import FSDPStrategy
from overwatch.overwatch import initialize_overwatch
from mm_utils.utils import *

# nohup bash scripts/pretrain_8_a100.sh > pretrain_8_a100.out 2>&1 &  3269779
# nohup bash scripts/grounded_8_a100.sh > grounded_8_a100.out 2>&1 &  1173215

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model', type=str, default='llava_next_video', choices=['llava_next_video'])
    parser.add_argument('--llm', type=str, default='llama3', choices=['llama3', 'vicuna'])

    parser.add_argument('--dataset', type=str, default='mix_pretrain', choices=['mix_pretrain', 'mix_grounded', 'mix_sft'])
    parser.add_argument('--max_txt_len', type=int, default=2048)
    parser.add_argument('--num_temporal_tokens', type=int, default=300)
    parser.add_argument('--num_frames', type=int, default=96)
    parser.add_argument('--num_segs', type=int, default=12)
    parser.add_argument('--stage', type=str, default="pretrain", choices=['pretrain', 'grounded', 'sft'])
    parser.add_argument('--lora', action='store_true')

    parser.add_argument('--sharding_strategy', type=str, default="full-shard", choices=['shard-grad-op', 'full-shard']) # shard-grad-op for pretrain, full-shard for SFT
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lora_lr', type=float, default=2e-4)
    parser.add_argument('--mm_proj_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3) # 1e-3 for pretrain, 2e-5 for SFT
    parser.add_argument('--global_batch_size', type=int, default=128)
    parser.add_argument('--per_device_batch_size', type=int, default=16)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--lr_scheduler_type', type=str, default="linear-warmup+cosine-decay")

    parser.add_argument('--save_dir', type=str, default='/data/hvw5451/weights/ckpt')
    parser.add_argument('--pretrained_proj', type=str, default='')

    args = parser.parse_args()
    return args

def worker_init_function(worker_id: int) -> None:
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_seq.generate_state(4))
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"
    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return worker_init_function if get_worker_init_fn else None

def pretrain(args) -> None:
    overwatch.info("VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    worker_init_fn = set_global_seed(args.seed, get_worker_init_fn=True)

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating VLM")

    if args.model == 'llava_next_video':
        from models.llava_next_video import LLAVA_NEXT_VIDEO
        model = LLAVA_NEXT_VIDEO(
            dtype=torch.bfloat16, 
            stage=args.stage, 
            max_txt_len=args.max_txt_len, 
            num_frames = args.num_frames,
            num_segs = args.num_segs,
            lora=args.lora,
            num_temporal_tokens=args.num_temporal_tokens,
            llm=args.llm,
            )

        model.vision_tower.to(model.dtype)
        model.video_encoder.to(model.dtype)
        model.multi_modal_projector.to(model.dtype)
        model.video_projecter.to(model.dtype)

        if args.stage=='grounded' and len(args.pretrained_proj) > 0:
            ckpt = torch.load(args.pretrained_proj, map_location='cpu')['model']
            model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
            model.video_projecter.load_state_dict(ckpt['video_projecter'])

    if overwatch.is_rank_zero():
        print(get_parameter_number(model))

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset")
    if args.dataset == 'mix_pretrain':
        from datasets.mix_pretrain import MixPretrain
        train_dataset = MixPretrain(
        webvid_anno_path = "/data/hvw5451/data/webvid-703k/filtered_train.json",
        webvid_video_path = "/data/hvw5451/data/webvid-703k/videos",
        panda_anno_path = "/data/hvw5451/data/panda70m_2m/simplified_panda.json",
        panda_video_path = "/data/hvw5451/data/panda70m_2m/clips",
        internvid_anno_path = "/data/hvw5451/data/internvid/simplified_internVid-10M-flt-filter.json",
        internvid_video_path = "/data/hvw5451/data/internvid/clips",
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='rand',
        llm=args.llm,
        )
    elif args.dataset == 'mix_grounded':
        from datasets.mix_grounded import MixGrounded
        train_dataset = MixGrounded(
        moment_anno_path = "/data/hvw5451/data/Moment-10m/simplified_GESM_data.json",
        moment_video_path = "/data/hvw5451/data/Moment-10m/videos",
        vtimellm_anno_path = "/data/hvw5451/data/vtimellm_stage2/simplified_train.json",
        vtimellm_video_path = '/data/hvw5451/data/vtimellm_stage2/clips',
        anet_anno_path = "/data/hvw5451/data/activitynet/captions/train.json",
        anet_video_path = '/data/hvw5451/data/activitynet/videos',
        internvidg_anno_path = "/data/hvw5451/data/InternVid-G/simplified_filter_train.json",
        internvidg_video_path = '/data/hvw5451/data/InternVid-G/videos',
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='rand',
        llm=args.llm,
        )
    elif args.dataset == 'mix_sft':
        from datasets.mix_sft import MixSFT
        train_dataset = MixSFT(
        anno_path = "/data/hvw5451/data/mix_sft/mix_sft.json",
        video_path = "/data/hvw5451/data",
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='rand',
        llm=args.llm,
        )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy")
    train_strategy = FSDPStrategy(
        args=args,
        vlm=model,
        device_id=device_id,
        epochs=args.epoch,
        max_steps=None,
        global_batch_size=args.global_batch_size,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        enable_gradient_checkpointing=True,
        enable_mixed_precision_training=True,
        reduce_in_full_precision=False,
        worker_init_fn=worker_init_fn,
        sharding_strategy=args.sharding_strategy,
    )

    train_strategy.run_setup(n_train_examples=len(train_dataset))

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, seed=args.seed)

    # Save ckpt
    overwatch.info("Save ckpt")
    train_strategy.save_checkpoint(run_dir=args.save_dir)

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Initialize Overwatch =>> Wraps `logging.Logger`
    overwatch = initialize_overwatch(__name__)
    args = parse_args()
    if overwatch.is_rank_zero():
        print(args)
    pretrain(args)