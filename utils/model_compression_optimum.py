import os
import clip
import torch

def _partition_visual_text_layers(state_dict : dict()):
    visual_layers = [item for item in list(state_dict.keys()) if item.startswith('visual')]
    text_layers   = [item for item in list(state_dict.keys()) if item.startswith('transformer')]
    
    return visual_layers, text_layers



def _multiply_SVD_matrices(r_layer_SVD: dict):
    """
    takes in a compressed layer and then 
    Args: 
        r_layer_SVD (dict): the layer dict with 'S','V','D' decomposed matrices
        
    Returns:
        torch.tensor after multiplying those tensors
    """

    r_recon = torch.matmul(r_layer_SVD['S'] * r_layer_SVD['V'], r_layer_SVD['D'].t())
    return r_recon

def _retrive_uncompress_matrix(r_model_dict: dict):
    """
    retrive the matrix from the compressed one

    Args:
        model_dict (dict): The model dictionary with 'S','V','D' parameters.

    Returns:
        dict: The model dictionary with original keys.
    """
    
    r_result_state_dict = {}
    for key, value in r_model_dict.items():
        if isinstance(value,dict)==True:
            r_result_state_dict[key] = _multiply_SVD_matrices(value)
        else:
            r_result_state_dict[key] = value
            
    return r_result_state_dict

def _determine_rank(r_dim_a : int, r_dim_b: int, r_compress_ratio: float):
    """
    determines the rank-k for achieveing the desired compression_ratio at each layer
    
    Args:
        dim_a (int): x-dimension
        dim_b (int): y-dimension
    
    Returns:
        int : rank-k
    
    """
    assert r_compress_ratio > 0 and r_compress_ratio < 1
    r_rank = int((r_compress_ratio*r_dim_a*r_dim_b)/(r_dim_a+r_dim_b+1))
    
    return r_rank

def _check_in_key(r_blocks : list, key : str):
    
    ans = True
    for item in r_blocks:
        if item not in key:
            ans = False
#             print(item,key)
            
    return ans

def for_7_attention_layers(key,n_layers):
    
    rslt = False
    for i in range(12):
        str_A = (f"visual.transformer.resblocks.{i}.attn")
#         visual.transformer.resblocks.1.mlp.c
        str_B = (f"visual.transformer.resblocks.{i}.mlp.c")
        if key.startswith(str_A) and i<n_layers:
            rslt = True
    return rslt

def _compress_model(r_model_dict :dict, r_compress_ratio :float, n_layers : int, d_visual: bool, d_text : bool, r_blocks : list, n_offset: int):
    """
    takes the modek dictionary and 
    
    Args:
        model_dict (dict): takes in model.state_dict() as parameters
        compress_ratio (float): takes in compression_ratio
        n_layers (int) : the layers from the end that needs to be compressed
        d_visual (bool) : 
    Returns:
        model_dict (dict): model.state_dict() with 3 additional keys:
                 -> 'S','V','D'
    """
    
    r_result_state_dict = {}
    idx = 0
    tot_layers = len(r_model_dict)
    layers_skipped = 0
#     gp_ratio = 0.95
#     list_keys = list(r_model_dict.keys())
#     list_keys.reverse()
#     for key_rev in list_keys:
#         key = key_rev
#         value = r_model_dict[key]
    last_idx_transformer = 0
    last_idx_vision = 0
    for key,value in r_model_dict.items():
        skip_layer = True
        
#         if _check_in_key(r_blocks,key) == False:
#             skip_layer = True
#         if d_text==True and d_visual==True:
#             skip_layer = False
    
#         if d_visual and key.startswith('visual')==False:
#             skip_layer = True
        
#         if d_text and key.startswith('transformer')==False:
#             skip_layer = True
    
        if for_7_attention_layers(key,n_layers):
            skip_layer = False
            
        if len(value.shape)==2 and not skip_layer:
            print(f"key-{key} gonna be compressed")
            layers_skipped += 1
            r_dim_a = value.shape[0]
            r_dim_b = value.shape[1]
            r_compress_ratio = r_compress_ratio#*gp_ratio
            r_rank = _determine_rank(r_dim_a, r_dim_b, r_compress_ratio)
#             print(r_rank,r_dim_a,r_dim_b)
            r_keys = ['S','V','D']
            r_values = list(torch.svd_lowrank(value.float(), q = r_rank, niter = 2, M=None))
#             print(r_values[0].shape,r_values[1].shape,r_values[2].shape)
            r_mod_layer = {k: v for k, v in zip(r_keys, r_values)}
            r_result_state_dict[key] = r_mod_layer
        else:
            r_result_state_dict[key] = value
        idx = idx + 1 
      
    return r_result_state_dict, layers_skipped