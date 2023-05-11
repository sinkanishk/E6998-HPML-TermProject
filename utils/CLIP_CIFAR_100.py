import os
import clip
import torch
from torchvision.datasets import CIFAR100

from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel

# model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# import cv2
import os
from tqdm import tqdm
import math
import numpy as np
import os
import clip
import torch
import re
import torch

model_b, preprocess = clip.load(params['model_size'], device='cpu')

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, images):
        self.images=images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        X = self.images[index]
        return preprocess(X)


def compute_model_performance(model_dict: dict()):
    device  = 'cuda'
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    params = {}
    params['model_size'] = "ViT-B/32"
    params['device'] = device if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(params['model_size'], device=params['device'])
    model.load_state_dict(model_dict)
    model = model.to('cuda')
    model.eval()
    model.requires_grad_(False)
    print("CLIP model loaded")
    params = {'batch_size': 32, 'shuffle': False, 'num_workers':4}
    testset=[]
    correctlabels=[]
    for image,class_id in cifar100:
        testset.append(image)
        correctlabels.append(class_id)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    text_features = model.encode_text(text_inputs).float()
    test_set = Dataset(testset)
    results = []
    testloader = torch.utils.data.DataLoader(test_set, **params)
    for idx, item in enumerate(tqdm(testloader,desc="Inference")):
        with torch.no_grad():
            item = item.to(device)
            outputs = model.encode_image(item).float()

        results.append(outputs)
    
    
    img_features = torch.vstack(results[0:-1]) 
    img_features /= img_features.norm(dim=-1, keepdim=True)
    values,indices = (img_features.squeeze() @ text_features.T).softmax(dim=1).topk(1)
    accuracy = {}
    for idx,item in enumerate(indices):
        accuracy[idx] = 0
        if (item.item() == correctlabels[idx]):
            accuracy[idx] = 1
    top1= sum(accuracy.values())/len(accuracy)
    print(f"Top-1 accuracy is {sum(accuracy.values())/len(accuracy)}")
    values,indices = (img_features.squeeze() @ text_features.T).softmax(dim=1).topk(5)
    accuracy = {}
    for idx,item in enumerate(indices):
        accuracy[idx] = 0
        for curr_item in item:
            if (curr_item.item() == correctlabels[idx]):
                accuracy[idx] = 1
    top5 = sum(accuracy.values())/len(accuracy)
    print(f"Top-5 accuracy is {sum(accuracy.values())/len(accuracy)}")
    return {"top-1":top1, "top-5":top5}