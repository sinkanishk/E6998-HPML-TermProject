import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
from imagenetv2_pytorch import ImageNetV2Dataset


def load_imagenet_classes():   
    import json
    with open('/home/ks4038_columbia_edu/hpml_project/E6998/E6998-HPML-TermProject/data/Imagenet.json','r') as f:
        imagenet_classes = json.load(f)['arr']
        
    imagenet_templates = [
        'itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.'
        ]
    
    return imagenet_templates, iamgenet_classes 
    
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights






def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def compute_model_performance(model_dict: dict()):
    model, preprocess = clip.load("ViT-B/32")
    model.load_state_dict(state_dict)
    model = model.to('cuda')
    images = ImageNetV2Dataset(transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
    imagenet_classes, imagenet_templates = load_imagenet_classes()
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    return {"top-1":top1, "top-5":top5}