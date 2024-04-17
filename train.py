import argparse
import torch
from torch import nn
from torchvision import datasets, models

from utils import data_transforms
from models import build_model, save_model, train_model, test_model
    
def main():
    parser:argparse.ArgumentParser = argparse.ArgumentParser(description="Training script with VGG16 or ResNet50")
    parser.add_argument("data_dir", type=str, help="Path to the directory with data")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save outputs")
    parser.add_argument("--arch", type=str, choices=["vgg16", "resnet50"], default="resnet50", help="Architecure type to use as base model")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate to use during training")
    parser.add_argument("--hidden_units", type=int, default=256, help="Number of hidden units to use in the head of the pretrained model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU training if availible")

    args:argparse.Namespace = parser.parse_args()
    gpu_availible:bool = torch.cuda.is_available()
    
    if args.gpu:
        if not gpu_availible:
            raise Exception("You have selected GPU training but no GPUs found on device, consider enabling gpu support or removing --gpu to train on CPU")    
        
        device = "cuda"
    else:
        device = "cpu"

    device = torch.device(device)

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
    }

    data_loaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64),
    } 

    model = build_model(args.arch, args.hidden_units)
    model, optimizer = train_model(device, args.arch, model, args.epochs, args.learning_rate, data_loaders)
    
    test_model(device, model, data_loaders)
    
    print(f"[INFO] saving model to directory {args.save_dir}")
    save_model(
        args.arch, 
        model, 
        optimizer,
        args.hidden_units,
        data_loaders["train"].batch_size, 
        image_datasets["train"].class_to_idx,
        args.save_dir    
    )

if __name__ == "__main__":
    main()