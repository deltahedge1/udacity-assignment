from torchvision import datasets, transforms, models
import torch
from torch import nn
import numpy as np
from PIL import Image

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.485, 0.456, 0.406]

train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(244),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

valid_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(244),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

data_transforms = {
    "train": train_transform,
    "test": valid_test_transform,
    "valid": valid_test_transform,
}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    # Define the transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image) 
    return image_tensor

def predict(device:str, image_path:str, model:nn.Module, checkpoint:dict, topk:int, cat_to_name:dict):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_tensor = process_image(image_path).unsqueeze(0).to(device)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(image_tensor)
        topk_logps, topk_idx = torch.topk(preds, topk)
        topk_probs, topk_idx = torch.exp(topk_logps).to("cpu").numpy().squeeze(), topk_idx.to("cpu").numpy().squeeze()
    
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx:_class for _class, idx in class_to_idx.items()}
    
    topk_class = [idx_to_class[idx] for idx in topk_idx]
    topk_labels = [cat_to_name[str(_class)] for _class in topk_class]
    
    return topk_labels, topk_probs