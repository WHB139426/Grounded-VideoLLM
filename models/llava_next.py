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
from typing import Callable, Dict, List, Optional, Type, Union
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from datasets.chat.base_template import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLaMA3_Template, Vicuna_Template
from models.modeling_llama import LlamaForCausalLM
from models.modeling_clip import CLIPVisionModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLAVA_NEXT(nn.Module):
    def __init__(
        self,
        dtype=torch.bfloat16,
        stage='pretrain',
        max_txt_len=2048,
        lora=False,
        llm='llama3',
    ):
        super().__init__()
        self.dtype = dtype
        self.max_txt_len = max_txt_len
        self.stage = stage
        self.lora = lora
        self.llm = llm

        if self.llm == 'llama3':
            self.config = AutoConfig.from_pretrained("/data3/whb/weights/llama3-llava-next-8b")
            self.tokenizer = AutoTokenizer.from_pretrained("/data3/whb/weights/Meta-Llama-3-8B-Instruct", use_fast=False, truncation_side="left")
            self.tokenizer.eos_token_id = 128009 # '<|eot_id|>'
            self.tokenizer.pad_token_id = 128001 # '<|end_of_text|>'
            self.separator = LLaMA3_Template.separator
        elif self.llm == 'vicuna':
            self.config = AutoConfig.from_pretrained("/data3/whb/weights/llava-v1.6-vicuna-7b")
            self.tokenizer = AutoTokenizer.from_pretrained("/data3/whb/weights/vicuna-7b-v1.5", use_fast=False, truncation_side="left")
            self.separator = Vicuna_Template.separator


        print("loading vision_tower")
        self.config.vision_config.torch_dtype = self.dtype
        self.vision_tower = CLIPVisionModel(self.config.vision_config)
        if self.llm == 'llama3':
            self.vision_tower.load_state_dict(torch.load('/data3/whb/weights/llama3-llava-next-8b-seperated/vision_model.pth', map_location='cpu'))
            self.image_newline = nn.Parameter(torch.load('/data3/whb/weights/llama3-llava-next-8b-seperated/image_newline.pth', map_location='cpu')['image_newline']).to(self.dtype)
        elif self.llm == 'vicuna':
            self.vision_tower.load_state_dict(torch.load('/data3/whb/weights/llava-v1.6-vicuna-7b-seperated/vision_model.pth', map_location='cpu'))
            self.image_newline = nn.Parameter(torch.load('/data3/whb/weights/llava-v1.6-vicuna-7b-seperated/image_newline.pth', map_location='cpu')['image_newline']).to(self.dtype)

        print("loading multi_modal_projector")
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)
        if self.llm == 'llama3':
            self.multi_modal_projector.load_state_dict(torch.load('/data3/whb/weights/llama3-llava-next-8b-seperated/multi_modal_projector.pth', map_location='cpu'))
        elif self.llm == 'vicuna':
            self.multi_modal_projector.load_state_dict(torch.load('/data3/whb/weights/llava-v1.6-vicuna-7b-seperated/multi_modal_projector.pth', map_location='cpu'))

        print("loading language_model")
        if self.llm == 'llama3':
            self.language_model = LlamaForCausalLM.from_pretrained('/data3/whb/weights/llama3-llava-next-8b-seperated/language_model_seperated', torch_dtype=self.dtype, use_cache=False)
        elif self.llm == 'vicuna':
            self.language_model = LlamaForCausalLM.from_pretrained('/data3/whb/weights/llava-v1.6-vicuna-7b-seperated/language_model_seperated', torch_dtype=self.dtype, use_cache=False)

        self.all_module_keys = ["vision_tower", "language_model", "multi_modal_projector"]

        if self.stage == 'pretrain':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen LLM")
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector"]
        elif self.stage == 'sft':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector", "language_model"]  
        
        if self.lora and self.stage == 'sft':
            from peft import get_peft_model, LoraConfig, TaskType
            print("LORA llm")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False if self.training else True, r=128, lora_alpha=256, lora_dropout=0.05, 
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", 'gate_proj'],
            )
            self.language_model = get_peft_model(self.language_model, peft_config)

            for name, module in self.language_model.named_modules(): 
                if "lora" in name.lower(): 
                    for param in module.parameters(): 
                        param.data = param.data.to(self.dtype)


    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_tower.get_fsdp_wrapping_policy()
        if self.lora:
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy_lora()
        else:
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LlavaMultiModalProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
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

    def encode_images(self, pixel_values):
        vision_feature_layer = -2
        vision_feature_select_strategy = "default"
        with self.maybe_autocast():
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:] # [bs, 576, 1024]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature 
            image_features = self.multi_modal_projector(selected_image_feature) # [bs, 576, 4096]
        image_newline = self.image_newline[None, None, :].expand(image_features.shape[0], 1, self.config.hidden_size).to(self.device)
        image_features = torch.cat([image_features, image_newline], dim=1) # [bs, 577, 4096]
        
        return image_features

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
            batch_image_features = self.encode_images(samples['pixel_values'])
            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['image_ids'])

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

        # image_features
        batch_image_features = self.encode_images(samples['pixel_values'])

        inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['image_ids'])

        with self.maybe_autocast():
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



# device = "cuda:3"
# # device = "cpu"
# dtype = torch.float32 if device == 'cpu' else torch.bfloat16
# llm = 'vicuna'

# image_processor = image_transform(image_size=336)
# pixel_values = torch.stack([image_processor(load_image('/data3/whb/frame_3.jpg')), image_processor(load_image('/data3/whb/frame_3.jpg'))], dim=0)

# if llm == 'llama3':
#     chat_template = LLaMA3_Template()
# elif llm == 'vicuna':
#     chat_template = Vicuna_Template()

# prompt_conversations = [
#     {"from": "human", "value": "<image>\n"+'Describe the image.'},
#     {"from": "gpt", "value": ''}
# ]
# sep, eos = chat_template.separator.apply()
# prompt = chat_template.encode(prompt_conversations).replace(eos, '')

# text_input_conversations = [
#     {"from": "human", "value": "<image>\n"+'Provide a one-sentence caption for the provided image.'},
#     {"from": "gpt", "value": 'This is an image.'},
#     {"from": "human", "value": "Can you show me more details?"},
#     {"from": "gpt", "value": "No problem."},
#     {"from": "human", "value": "What can i say."},
#     {"from": "gpt", "value": "Mamba out."},
# ]
# text_input = chat_template.encode(text_input_conversations)

# prompt_conversations_copy = [
#     {"from": "human", "value": "<image>\n"+'How many peopole are there?'},
#     {"from": "gpt", "value": ''}
# ]
# sep, eos = chat_template.separator.apply()
# prompt_copy = chat_template.encode(prompt_conversations_copy).replace(eos, '')


# text_input_conversations_copy = [
#     {"from": "human", "value": "<image>\n"+'Show me the truth'},
#     {"from": "gpt", "value": 'I cannot.'},
#     {"from": "human", "value": "Can you show me more details?"},
#     {"from": "gpt", "value": "This is a sad story."},
# ]
# text_input_copy = chat_template.encode(text_input_conversations_copy)




# model = LLAVA_NEXT(dtype=dtype, llm=llm)
# model.to(device)
# print(get_parameter_number(model))

# samples = {
#         "image_ids": ['xxxx', 'xxxx'],
#         "text_inputs": [text_input, text_input_copy],
#         "prompts": [prompt, prompt_copy],
#         "pixel_values": pixel_values.to(device),
#     }

# loss = model(samples)
# print(loss)

# with torch.inference_mode():
#     generate_kwargs = {
#         "do_sample": False,
#         "num_beams": 1, 
#         "max_new_tokens": 256,
#         "temperature":1,
#         }
#     output_text = model.generate(samples, **generate_kwargs)

# print(samples['text_inputs'])
# print(samples['prompts'])
# print(output_text)
