#!/usr/bin/env python
# coding: utf-8
# %%

# %%
from utils.model_compression import _compress_model, _retrive_uncompress_matrix
import torch
import clip
import json

# %%

"""
load original clip model
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# %%


def compression_tot(clip_compress_state_dict,state_dict):
    total_params = 0
    for param_tensor in state_dict.values():
        total_params += param_tensor.numel()
    
    total_params_comp = 0
    for param_tensor in clip_compress_state_dict.values():
        if isinstance(param_tensor,dict)==True:
            for key,value in param_tensor.items():
                total_params_comp += value.numel()
        else:
            total_params_comp += param_tensor.numel()
    
    return (total_params - total_params_comp)/total_params


# %%
from utils.CLIP_ImageNet import compute_model_performance
# CLIP_CIFAR_100
# from utils.CLIP_CIFAR_100 import compute_model_performance

state_dict = model.state_dict()
compress_ratio =  [0.1+0.1*i for i in range(9)]
# n_back
# compress_ratio
# n_ratio.reverse()


# %%

path = '/home/ks4038_columbia_edu/hpml_project/E6998/E6998-HPML-TermProject/results/'


# %%

# Attention
# Attention with IMAGENET
r_blocks = ['.resblocks.1.attn.']
n_layers = [i+1 for i in range(9)]
n_layers = 9
n_prune = [i for i in range(0,6)]
result = {}
for ratio in compress_ratio:
    result[ratio] = {}
    for prune in n_prune:
        result[ratio][prune] = {}
        clip_compress_state_dict, layers_skipped = _compress_model(state_dict, ratio, n_layers, True, False, r_blocks,0)
        clip_uncompress_state_dict = _retrive_uncompress_matrix(clip_compress_state_dict)
        print(compression_tot(clip_compress_state_dict,state_dict))
        result[ratio][prune]['perf'] = compute_model_performance(clip_uncompress_state_dict, prune)
        result[ratio][prune]['comp'] = compression_tot(clip_compress_state_dict,state_dict)
        print(result)
        with open(path+'vision_only_from_start_without_attention.json','w') as f:
            json.dump(result,f,indent=4)


# %%




