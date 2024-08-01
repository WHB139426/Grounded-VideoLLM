from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import safetensors
import torch
import sys
import os
from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub import hf_hub_url
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.modeling_llama import LlamaForCausalLM


# repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# save_path = '/data3/whb/weights/ Meta-Llama-3-8B-Instruct'

# login()
# api = HfApi()

# from huggingface_hub import snapshot_download

# snapshot_download(
#   repo_id=repo_id,
#   local_dir=save_path,
#   local_dir_use_symlinks=False,
# )


def st2ckpt():
    state_dict = {}
    # 加载 .safetensors 文件
    data_1 = safetensors.torch.load_file('/data3/whb/weights/llama3-llava-next-8b/model-00001-of-00004.safetensors')
    data_2 = safetensors.torch.load_file('/data3/whb/weights/llama3-llava-next-8b/model-00002-of-00004.safetensors')
    data_3 = safetensors.torch.load_file('/data3/whb/weights/llama3-llava-next-8b/model-00003-of-00004.safetensors')
    data_4 = safetensors.torch.load_file('/data3/whb/weights/llama3-llava-next-8b/model-00004-of-00004.safetensors')
    state_dict.update(data_1)
    state_dict.update(data_2)
    state_dict.update(data_3)
    state_dict.update(data_4)

    image_newline_dic = {}
    vision_tower_dic = {}
    mlp_dic = {}
    language_model_dic = {}

    for key in state_dict:
        if 'model.vision_tower.vision_tower' in key:
            temp_key = key.replace('model.vision_tower.vision_tower.','')
            vision_tower_dic.update({temp_key: state_dict[key].to(torch.float32)})
        elif 'model.mm_projector' in key:
            if key == 'model.mm_projector.0.bias':
                temp_key = 'linear_1.bias'
            elif key == 'model.mm_projector.0.weight':
                temp_key = 'linear_1.weight'
            elif key == 'model.mm_projector.2.bias':
                temp_key = 'linear_2.bias'
            elif key == 'model.mm_projector.2.weight':
                temp_key = 'linear_2.weight'
            mlp_dic.update({temp_key: state_dict[key].to(torch.float32)})
        elif 'image_newline' in key:
            image_newline_dic.update({'image_newline': state_dict[key].to(torch.float32)})
        else:
            language_model_dic.update({key: state_dict[key]})


    return state_dict, vision_tower_dic, mlp_dic, image_newline_dic, language_model_dic


state_dict, vision_tower_dic, mlp_dic, image_newline_dic, language_model_dic = st2ckpt()



torch.save(vision_tower_dic, '/data3/whb/weights/llama3-llava-next-8b-seperated/vision_model.pth')
torch.save(mlp_dic, '/data3/whb/weights/llama3-llava-next-8b-seperated/multi_modal_projector.pth')
torch.save(image_newline_dic, '/data3/whb/weights/llama3-llava-next-8b-seperated/image_newline.pth')
torch.save(language_model_dic, '/data3/whb/weights/llama3-llava-next-8b-seperated/language_model.pth')

model = LlamaForCausalLM(config=AutoConfig.from_pretrained('/data3/whb/weights/Meta-Llama-3-8B-Instruct'))
model.load_state_dict(torch.load('/data3/whb/weights/llama3-llava-next-8b-seperated/language_model.pth', map_location='cpu'), strict=True)
model.save_pretrained('/data3/whb/weights/llama3-llava-next-8b-seperated/language_model_seperated/')