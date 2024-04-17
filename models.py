from pathlib import Path

from collections import OrderedDict
import torch
from torch import nn, optim
from torchvision import models

# model metadata to build the models
supported_models={
    "resnet50":{
        "in_features":8192,
        "head_name":"fc",
        "model": models.resnet50
    },
    "vgg16":{
        "in_features":25088,
        "head_name":"classifier",
        "model": models.vgg16
    }
}

def build_model(pretrained_model_name:str, hidden_layer_1:int) -> nn.Module:
    """
    build the model architecture which can be used for training or for loading from a checkpoint
    """
    model_metadata:nn.Module = supported_models[pretrained_model_name]
    model:nn.Module = model_metadata["model"](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model_head:nn.Module = nn.Sequential(
        OrderedDict(([
            ("fc1", nn.Linear(model_metadata["in_features"], hidden_layer_1)),
            ("relu", nn.ReLU()),
            ("dropput", nn.Dropout(p=0.2)),
            ("fc2", nn.Linear(hidden_layer_1, 102)),
            ("output", nn.LogSoftmax(dim=1))
        ]))
    )

    model._modules[model_metadata["head_name"]] = model_head

    return model

def save_model(base_arch:str, model:nn.Module, optimizer, hidden_layer_1:int, batch_size:int, class_to_idx:dict, save_dir:str):
    checkpoint = {
        "base_model": base_arch,
        'input_size': [3, 224, 224],
        'batch_size': batch_size,
        'hidden_layer_1': hidden_layer_1,
        'output_size': 102,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }

    output_path = Path(save_dir) / f"{base_arch}_checkpoint.pth"
    torch.save(checkpoint, str(output_path))

def train_model(device:str, arch:str, model:torch.nn.Module, epochs:int, lr:float, data_loaders:dict, weight_decay:float=1e-4, print_every:int=5) -> torch.nn.Module:
    print(f"INFO[log]: using device: {device}")
    
    criterion = nn.NLLLoss()
    if arch == "vgg16":
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    elif arch == "resnet50":
        optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    
    total_steps=len(data_loaders["train"]) 
    running_loss=0
    
    print(f"INFO[log]: starting to train model, total epochs: {epochs}")
    
    for epoch in range(epochs):
        steps = 0
        epoch += 1

        for inputs, labels in data_loaders["train"]:
            model.train()
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    testloader = data_loaders["valid"] #NEED TO RETRAIN FOR THIS
                    
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch}/{epochs}.. "
                  f"Step {steps}/{total_steps}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                
                running_loss = 0
            
    return model, optimizer

def test_model(device:str, model:nn.Module, data_loaders) -> None:
    criterion = nn.NLLLoss()
    test_loss = 0
    test_accuracy= 0
    
    with torch.no_grad():
        testloader = data_loaders["test"]
        
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)  # Use the 'm' variable consistently for the model
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"[INFO] Test loss: {test_loss/len(testloader):.3f}.. "
          f"[INFO] Test accuracy: {test_accuracy/len(testloader):.3f}")

def build_from_chpt(device, path):

    try:
        checkpoint = torch.load(path, map_location=device)
    except:
        checkpoint = torch.load(path)
        
    pretrained_model_name = checkpoint["base_model"]
    hidden_layer_1 = checkpoint["hidden_layer_1"]
    
    model = build_model(pretrained_model_name, hidden_layer_1)
    model.load_state_dict(checkpoint["model_state"])

    return model, checkpoint