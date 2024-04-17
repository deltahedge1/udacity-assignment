import argparse
import json
import torch
from utils import process_image, predict
from models import build_from_chpt

def main():
    parser:argparse.ArgumentParser = argparse.ArgumentParser(description="Predict topk from an image")
    parser.add_argument("image_path", type=str, help="path to image to predict topk flowers")
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="path to the category names in a json")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict for the image")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU training if availible")
    
    args:argparse.Namespace = parser.parse_args()
    gpu_availible = torch.cuda.is_available()
    
    if args.gpu:
        if not gpu_availible:
            raise Exception("You have selected GPU training but no GPUs found on device, consider enabling gpu support or removing --gpu to train on CPU")    
          
        device = "cuda"
    else:
        device = "cpu"
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model, checkpoint = build_from_chpt(device, args.checkpoint)
    topk_labels, topk_probs = predict(device, args.image_path, model, checkpoint, args.top_k, cat_to_name)
    
    for label, prob  in zip(topk_labels, topk_probs):
        print(f"{label}: {prob*100:.2f}%")
    
if __name__ == "__main__":
    main()