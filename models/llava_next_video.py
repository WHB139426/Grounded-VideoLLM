import sys
import os
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import copy
from transformers import AutoTokenizer, AutoConfig
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from functools import partial
import einops
import math
from einops import rearrange, repeat
from typing import Callable, Dict, List, Optional, Type, Union
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from datasets.chat.base_template import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLaMA3_Template, Vicuna_Template
from models.modeling_llama import LlamaForCausalLM
from models.modeling_clip import CLIPVisionModel
from models.internvideo2 import pretrain_internvideo2_1b_patch14_224, interpolate_pos_embed_internvideo2_new
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Video_Projecter(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.intermediate_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class LLAVA_NEXT_VIDEO(nn.Module):
    def __init__(
        self,
        dtype=torch.bfloat16,
        stage='pretrain',
        max_txt_len=2048,
        num_frames=96,
        num_segs=12,
        lora=False,
        num_temporal_tokens=300,
        llm='llama3',
    ):
        super().__init__()
        self.dtype = dtype
        self.max_txt_len = max_txt_len
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.stage = stage
        self.lora = lora
        self.num_temporal_tokens = num_temporal_tokens
        self.llm = llm

        if self.llm == 'llama3':
            self.config = AutoConfig.from_pretrained("/data/hvw5451/weights/llama3-llava-next-8b")
            self.tokenizer = AutoTokenizer.from_pretrained("/data/hvw5451/weights/Meta-Llama-3-8B-Instruct", use_fast=False, truncation_side="left")
            self.tokenizer.eos_token_id = 128009 # '<|eot_id|>'
            self.tokenizer.pad_token_id = 128001 # '<|end_of_text|>'
            self.separator = LLaMA3_Template.separator
        elif self.llm == 'vicuna':
            self.config = AutoConfig.from_pretrained("/data/hvw5451/weights/llava-v1.6-vicuna-7b")
            self.tokenizer = AutoTokenizer.from_pretrained("/data/hvw5451/weights/vicuna-7b-v1.5", use_fast=False, truncation_side="left")
            self.separator = Vicuna_Template.separator

        print("loading vision_tower")
        self.config.vision_config.torch_dtype = self.dtype
        self.vision_tower = CLIPVisionModel(self.config.vision_config)
        if self.llm == 'llama3':
            self.vision_tower.load_state_dict(torch.load('/data/hvw5451/weights/llama3-llava-next-8b-seperated/vision_model.pth', map_location='cpu'))
            self.image_newline = torch.load('/data/hvw5451/weights/llama3-llava-next-8b-seperated/image_newline.pth', map_location='cpu')['image_newline'].to(self.dtype)
        elif self.llm == 'vicuna':
            self.vision_tower.load_state_dict(torch.load('/data/hvw5451/weights/llava-v1.6-vicuna-7b-seperated/vision_model.pth', map_location='cpu'))
            self.image_newline = torch.load('/data/hvw5451/weights/llava-v1.6-vicuna-7b-seperated/image_newline.pth', map_location='cpu')['image_newline'].to(self.dtype)

        print("loading video_encoder")
        self.video_encoder = pretrain_internvideo2_1b_patch14_224(self.num_frames//self.num_segs)
        state_dict = torch.load('/data/hvw5451/weights/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt', map_location='cpu')
        interpolate_pos_embed_internvideo2_new(state_dict, self.video_encoder, orig_t_size=4)
        self.video_encoder.load_state_dict(state_dict, strict=True)

        print("loading multi_modal_projector")
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)
        if self.llm == 'llama3':
            self.multi_modal_projector.load_state_dict(torch.load('/data/hvw5451/weights/llama3-llava-next-8b-seperated/multi_modal_projector.pth', map_location='cpu'))
        elif self.llm == 'vicuna':
            self.multi_modal_projector.load_state_dict(torch.load('/data/hvw5451/weights/llava-v1.6-vicuna-7b-seperated/multi_modal_projector.pth', map_location='cpu'))

        print("loading video_projector")
        self.video_projecter = Video_Projecter(1408, self.config.hidden_size)

        print("loading language_model")
        if self.llm == 'llama3':
            self.language_model = LlamaForCausalLM.from_pretrained('/data/hvw5451/weights/llama3-llava-next-8b-seperated/language_model_seperated', torch_dtype=self.dtype, use_cache=False)
        elif self.llm == 'vicuna':
            self.language_model = LlamaForCausalLM.from_pretrained('/data/hvw5451/weights/llava-v1.6-vicuna-7b-seperated/language_model_seperated', torch_dtype=self.dtype, use_cache=False)

        self.all_module_keys = ["vision_tower", "language_model", "video_encoder", "multi_modal_projector", "video_projecter"]

        if self.stage == 'pretrain':
            print("Frozen vision_tower")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False
            print("Frozen LLM")
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector", "video_projecter"]

        elif self.stage == 'grounded':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False

            self.reset_embeddings()

            if self.lora:
                print("LORA llm")
                self.lora_model()

            print("Frozen Part LLM")
            for name, param in self.language_model.named_parameters():
                if 'lm_head' in name or 'embed_tokens' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector", "video_projecter", "language_model"]

        elif self.stage == 'sft':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False

            self.reset_embeddings()

            if self.lora:
                print("LORA llm")
                self.lora_model()

            print("Frozen Part LLM")
            for name, param in self.language_model.named_parameters():
                if 'lm_head' in name or 'embed_tokens' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.trainable_module_keys = ["multi_modal_projector", "video_projecter", "language_model"]  
        
    def lora_model(self,):
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False if self.training else True, r=128, lora_alpha=256, lora_dropout=0.05, 
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", 'gate_proj'],
        )
        self.language_model = get_peft_model(self.language_model, peft_config)

        for name, module in self.language_model.named_modules(): 
            if "lora" in name.lower(): 
                for param in module.parameters(): 
                    param.data = param.data.to(self.dtype)

    def reset_embeddings(self,):
        """
        tokenizer and embed
        """
        special_token_list = [f'<{i}>' for i in range(self.num_temporal_tokens + 1)]
        self.tokenizer.add_tokens(special_token_list)
        num_new_tokens = len(special_token_list)
        self.language_model.config.vocab_size = len(self.tokenizer)

        """
        word embeddings
        """
        embedding_layer = self.language_model.get_input_embeddings()
        average_embedding = torch.mean(embedding_layer.weight, dim=0)

        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape
        
        new_embeddings = nn.Embedding(old_num_tokens + num_new_tokens, old_embedding_dim)
        new_embeddings.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[:old_num_tokens, :]
        new_embeddings.weight.data[old_num_tokens:, :] = average_embedding

        self.language_model.set_input_embeddings(new_embeddings)

        """
        lm_head
        """
        lm_head = self.language_model.get_output_embeddings()
        average_head = torch.mean(lm_head.weight, dim=0)

        old_num_tokens, old_hidden_size = lm_head.weight.shape

        new_lm_head = nn.Linear(old_hidden_size, old_num_tokens + num_new_tokens)
        new_lm_head.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
        new_lm_head.weight.data[:old_num_tokens, :] = lm_head.weight.data[:old_num_tokens, :]
        new_lm_head.weight.data[old_num_tokens:, :] = average_head

        self.language_model.set_output_embeddings(new_lm_head)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_tower.get_fsdp_wrapping_policy()
        video_fsdp_wrapping_policy = self.video_encoder.get_fsdp_wrapping_policy()
        if self.stage == 'grounded':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy_embedding()
        elif self.stage=='pretrain':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy()
        elif self.stage == 'sft':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy_embedding()
        else:
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LlavaMultiModalProjector, Video_Projecter}, # Video_Projecter
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                video_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = 1
        if self.llm == 'llama3':
            labels, cur_len = self._make_masks_llama3(labels, tokenizer, sep, eos_token_length, rounds)
        elif self.llm == 'vicuna':
            labels, cur_len = self._make_masks_vicuna(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len != total_len:
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            )
        return labels
        
    def _make_masks_llama3(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1 # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - bos_token_length
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    def _make_masks_vicuna(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1 # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length
            if i >=1:
                instruction_len -= 1
                round_len -= 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
        
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).to(self.device)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def prepare_batch(self, text_inputs):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for text in text_inputs:
            input_ids = self.tokenizer_image_token(text, self.tokenizer, return_tensors='pt')
            labels = self.make_labels(input_ids, text, self.tokenizer).to(self.device)
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long).to(self.device)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # Pad the sequences
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)

        # Truncate the sequences
        if batch_input_ids.shape[1] > self.max_txt_len:
            batch_input_ids = batch_input_ids[:, :self.max_txt_len]
            batch_labels = batch_labels[:, :self.max_txt_len]
            batch_labels[:, -1] = self.tokenizer.eos_token_id
            batch_attention_mask = batch_attention_mask[:, :self.max_txt_len]

        # for labels in batch_labels:
        #     print(labels)
        #     test_labels = []
        #     for i in labels:
        #         if i != -100:
        #             test_labels.append(self.tokenizer.decode(torch.tensor([i]), skip_special_tokens=False))
        #         else:
        #             test_labels.append(-100)
        #     print(test_labels)

        return batch_input_ids, batch_labels, batch_attention_mask

    def interpolated_position_embedding(self, samples):
        if self.llm == 'llama3':
            skip_len = 1
        elif self.llm == 'vicuna':
            skip_len = 2

        time_pos_left_tokens = self.tokenizer(samples['time_pos_left'], return_tensors="pt").input_ids[:, skip_len:].to(self.device) # [bs, num_frames], without BOS token
        time_pos_right_tokens = self.tokenizer(samples['time_pos_right'], return_tensors="pt").input_ids[:, skip_len:].to(self.device) # [bs, num_frames], without BOS token
        coefficient_left = samples['coefficient_left'] # [bs, num_frames]
        coefficient_right = samples['coefficient_right'] # [bs, num_frames]
        time_pos_left_embeds = self.get_input_embeddings()(time_pos_left_tokens) # [bs, num_frames, 4096]
        time_pos_right_embeds = self.get_input_embeddings()(time_pos_right_tokens) # [bs, num_frames, 4096]

        time_pos_left_embeds = time_pos_left_embeds*coefficient_left.unsqueeze(-1) # [bs, num_frames, 4096]
        time_pos_right_embeds = time_pos_right_embeds*coefficient_right.unsqueeze(-1) # [bs, num_frames, 4096]
        interpolated_time_pos_embeds = time_pos_left_embeds + time_pos_right_embeds # [bs, num_frames, 4096]

        return interpolated_time_pos_embeds

    def encode_images(self, samples):

        spatial_pixel_values = samples['spatial_pixel_values'] # [bs, num_segs, 3, 336, 336]
        temporal_pixel_values = samples['temporal_pixel_values'] # [bs, num_frames, 3, 224, 224]

        batch_size, num_segs, _, _, _ = spatial_pixel_values.shape
        batch_size, num_frames, _, _, _ = temporal_pixel_values.shape
        num_frames_per_seg = num_frames//num_segs

        """
        image features
        """
        spatial_pixel_values = rearrange(spatial_pixel_values, "b t c h w -> (b t) c h w") # [bs*num_frames, 3, 336, 336]
        image_outputs = self.vision_tower(spatial_pixel_values, output_hidden_states=True)
        image_features = image_outputs.hidden_states[-2][:, 1:] # [bs*num_frames, 576, 1024]

        # Aadptive Pooling for image features
        def convert_Fembeddings2video(input, num_videos, frame_shape):
            input = einops.rearrange(input, 
                                    '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                    num_videos=num_videos, h=frame_shape[0])
            return input
        frame_shape = (int(math.sqrt(image_features.shape[1])), int(math.sqrt(image_features.shape[1]))) # [24, 24] 
        hidden_states = convert_Fembeddings2video(image_features, batch_size, frame_shape) # [bs, 1024, num_segs, 24, 24] 
        hidden_states = nn.AdaptiveAvgPool3d([num_segs, 8, 8])(hidden_states) # [bs, 1024, num_segs, 8, 8]  
        image_features = einops.rearrange(hidden_states, 'batch_size_num_videos embed_dims num_frames h w -> batch_size_num_videos num_frames (h w) embed_dims', )
        image_features = self.multi_modal_projector(image_features) # [bs, num_segs, 64, 4096] 

        """
        segment features
        """
        temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs (num_segs num_frames_per_seg) c h w -> bs num_segs num_frames_per_seg c h w', num_segs=num_segs)
        temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs num_segs num_frames_per_seg c h w -> (bs num_segs) c num_frames_per_seg h w')
        segment_features = self.video_encoder(temporal_pixel_values, None, False, x_vis_return_idx=-2, x_vis_only=True)[:, 1:, :] # [bs*num_segs, num_frames_per_seg*256, 1408]  
        segment_features = einops.rearrange(segment_features, 'bs_num_segs (num_frames_per_seg hw) d -> bs_num_segs num_frames_per_seg hw d', num_frames_per_seg=num_frames_per_seg) # [bs*num_segs, num_frames_per_seg, 256, 1408] 

        # Aadptive Pooling for segment features
        frame_shape = (int(math.sqrt(segment_features.shape[2])), int(math.sqrt(segment_features.shape[2]))) # [16, 16] 
        hidden_states = einops.rearrange(segment_features, 'bs_num_segs num_frames_per_seg (h w) d -> bs_num_segs d num_frames_per_seg h w', h=frame_shape[0]) # [bs*num_segs, 1408, num_frames_per_seg, 16, 16]
        pool_size = 4
        hidden_states = nn.AdaptiveAvgPool3d([num_frames_per_seg, pool_size, pool_size])(hidden_states) # [bs*num_segs, 1408, num_frames_per_seg, 4, 4]  
        segment_features = einops.rearrange(hidden_states, '(bs num_segs) d num_frames_per_seg h w -> bs num_segs num_frames_per_seg (h w) d', num_segs=num_segs) # [bs, num_segs, num_frames_per_seg, 16, 1408]
        segment_features = einops.rearrange(segment_features, 'bs num_segs num_frames_per_seg hw d -> bs num_segs (num_frames_per_seg hw) d') # [bs, num_segs, num_frames_per_seg*16, 1408]
        segment_features = self.video_projecter(segment_features) # [bs, num_segs, num_frames_per_seg*16, 4096]

        # if self.stage == 'grounded' or self.stage == 'sft':
        #     interpolated_time_pos_embeds = self.interpolated_position_embedding(samples) # [bs, num_frames, 4096]
        #     segment_features = einops.rearrange(segment_features, 'bs num_segs (num_frames_per_seg hw) d -> bs (num_segs num_frames_per_seg) hw d', num_frames_per_seg=num_frames_per_seg, hw=pool_size*pool_size) # [bs, num_frames, 16, 4096]
        #     segment_features += interpolated_time_pos_embeds.unsqueeze(2)
        #     segment_features = einops.rearrange(segment_features, 'bs (num_segs num_frames_per_seg) hw d -> bs num_segs (num_frames_per_seg hw) d', num_segs=num_segs, num_frames_per_seg=num_frames_per_seg, hw=pool_size*pool_size) # [bs, num_segs, num_frames_per_seg*16, 4096]

        """
        video features
        """
        image_newline = self.image_newline[None, None, None, :].expand(batch_size, num_segs, 1, self.config.hidden_size).to(self.device)
        video_features = torch.cat([image_features, segment_features, image_newline], dim=2).to(self.device) # [bs, num_segs, 64+128+1, 4096]
        video_features = einops.rearrange(video_features, 'bs num_segs seq_len d -> bs (num_segs seq_len) d')

        return video_features

    def prepare_multimodal_inputs(self, batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, batch_image_ids):
        new_input_embeds = []
        new_labels = []
        new_attention_masks = []
        for image_embeds, input_ids, labels, attention_mask, image_ids in zip(batch_image_features, batch_input_ids, batch_labels, batch_attention_mask, batch_image_ids):
            """
            image_embeds: [576, 4096]
            input_ids: [seq]
            labels: [seq]
            attention_mask: [seq]
            """
            image_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
            pre_embeds = self.get_input_embeddings()(input_ids[:image_index].unsqueeze(0))[0]
            post_embeds = self.get_input_embeddings()(input_ids[image_index+1:].unsqueeze(0))[0]

            if image_ids == 'text':
                new_input_embeds.append(torch.cat([pre_embeds, post_embeds, image_embeds], dim=0))
                new_labels.append(torch.cat([labels[:image_index], labels[image_index+1:], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device)*IGNORE_INDEX], dim=0))
                new_attention_masks.append(torch.cat([attention_mask[:image_index], attention_mask[image_index+1:], torch.zeros(image_embeds.shape[0], dtype=torch.long).to(self.device)], dim=0))
            else:
                new_input_embeds.append(torch.cat([pre_embeds, image_embeds, post_embeds], dim=0))
                new_labels.append(torch.cat([labels[:image_index], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device)*IGNORE_INDEX, labels[image_index+1:]], dim=0))
                new_attention_masks.append(torch.cat([attention_mask[:image_index], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device), attention_mask[image_index+1:]], dim=0))

        new_input_embeds = torch.stack(new_input_embeds, dim=0).to(self.device)
        new_labels = torch.stack(new_labels, dim=0).to(self.device)
        new_attention_masks = torch.stack(new_attention_masks, dim=0).to(self.device)

        return new_input_embeds, new_labels, new_attention_masks

    def forward(
        self,
        samples,
    ):
        with self.maybe_autocast():
            batch_input_ids, batch_labels, batch_attention_mask = self.prepare_batch(samples['text_inputs'])
            batch_image_features = self.encode_images(samples)
            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['video_ids'])

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                return_dict=True,
                labels=labels,
            )
        loss = outputs.loss
        return {"loss": loss}

    @torch.inference_mode()
    def generate(
        self,
        samples,
        **generate_kwargs
    ):
        prompts = samples['prompts']
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for text in prompts:
            input_ids = self.tokenizer_image_token(text, self.tokenizer, return_tensors='pt')
            labels = copy.deepcopy(input_ids)
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long).to(self.device)
            batch_input_ids.append(torch.flip(input_ids, dims=[0])) # reverse the sequence
            batch_labels.append(labels)
            batch_attention_mask.append(torch.flip(attention_mask, dims=[0])) # reverse the sequence
        
        # Pad the sequences
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)

        # Truncate the sequences
        if batch_input_ids.shape[1] > self.max_txt_len:
            batch_input_ids = batch_input_ids[:, :self.max_txt_len]
            batch_labels = batch_labels[:, :self.max_txt_len]
            batch_attention_mask = batch_attention_mask[:, :self.max_txt_len]

        # Reverse the sequence back
        batch_input_ids = torch.flip(batch_input_ids, dims=[1])
        batch_attention_mask = torch.flip(batch_attention_mask, dims=[1])

        with self.maybe_autocast():
            # image_features
            batch_image_features = self.encode_images(samples)

            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['video_ids'])

            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_masks,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

# device = "cuda:7"
# # device = "cpu"
# dtype = torch.float32 if device == 'cpu' else torch.bfloat16
# num_frames=96
# num_segs=12
# stage='grounded'
# llm = 'vicuna'

# from datasets.mix_grounded import MixGrounded
# from torch.utils.data import Dataset, DataLoader
# dataset = MixGrounded(llm=llm)
# data_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=4)

# model = LLAVA_NEXT_VIDEO(dtype=dtype, stage=stage, num_frames=num_frames, num_segs=num_segs, llm=llm)
# model.to(device)
# print(get_parameter_number(model))

# llama3_prompt = "<|start_header_id|>system<|end_header_id|>You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|start_header_id|>user<|end_header_id|><image>\nShare a concise interpretation of the video provided.<|start_header_id|>assistant<|end_header_id|>"
# vicuna_prompt = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\nUSER:<image>\nShare a concise interpretation of the video provided.\nASSISTANT:"

# if llm == 'llama3':
#     prompt = llama3_prompt
# elif llm == 'vicuna':
#     prompt = vicuna_prompt

# for step, data in enumerate(data_loader):

#     samples = {
#             "video_ids": data['video_ids'],
#             "text_inputs": data['text_inputs'],
#             "prompts": [prompt for i in range(len(data['text_inputs']))],
#             "temporal_pixel_values":data['temporal_pixel_values'].to(device),
#             "spatial_pixel_values": data['spatial_pixel_values'].to(device),
#         }
#     loss = model(samples)
#     print(loss)

#     with torch.inference_mode():
#         generate_kwargs = {
#             "do_sample": False,
#             "num_beams": 1, 
#             "max_new_tokens": 256,
#             "temperature":1,
#             }
#         output_text = model.generate(samples, **generate_kwargs)

#     print("----------------------")
#     print(samples['video_ids'])
#     print(samples['text_inputs'])
#     print(samples['prompts'])
#     print(output_text)

#     if step==5:
#         break









