import torch
import wandb
from time import perf_counter

import sys
sys.path.insert(0, './models')

from models.nasnet import NASNetALarge
from models.mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from models.MnasNet import MnasNet
from models.mobilenet import MobileNetV3
from efficientnet_pytorch import EfficientNet

def get_model_info(model, name):
    model.eval()
    model.to('cuda:0')

    wandb.init(project='sysdl-term-project', name=f'{name}-inference' )
    wandb.watch(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters())
    
    inference_runs = 20
    
    if name == 'Inception-v3':
        input_tensor = torch.rand(32, 3, 299, 299).to('cuda:0')
    else:
        input_tensor = torch.rand(32, 3, 224, 224).to('cuda:0')
    
    start_time = perf_counter()
    for _ in range(inference_runs):
        out = model(input_tensor)
    inference_latency = perf_counter() - start_time
    
    model_info = {
        'Name': name,
        'Trainable Parameters': trainable_params,
        'Total Parameters': total_params,
        'Inference Latency': inference_latency/inference_runs,
        'GPU Name': torch.cuda.get_device_name()
    }
    
    return model_info

def get_NASNetALarge_info():
    nasnet = NASNetALarge()
    model_info = get_model_info(nasnet, 'NASNet-A Large')
    print(model_info)

def get_FBNetV2_info():
    FBNetv2 = fbnet('dmasking_f4')
    model_info = get_model_info(FBNetv2, name='FBNet-v2')
    print(model_info)

def get_MNasNet_info():
    mnasnet = MnasNet()
    model_info = get_model_info(mnasnet, name='MNasNet')
    print(model_info)

def get_InceptionV3_info():
    inceptionv3 = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False)
    model_info = get_model_info(inceptionv3, name='Inception-v3')
    print(model_info)

def get_EfficientNetB7_info():
    efficientnetb7 = EfficientNet.from_name('efficientnet-b7')
    model_info = get_model_info(efficientnetb7, name='EfficientNetB7')
    print(model_info)

def get_EfficientNetB0_info():
    efficientnetb0 = EfficientNet.from_name('efficientnet-b0')
    model_info = get_model_info(efficientnetb0, name='EfficientNetB0')
    print(model_info)

def get_ResNeXt101_info():
    resnext_101 = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=False)
    model_info = get_model_info(resnext_101, name='ResNeXt-101')
    print(model_info)

def get_MobileNetV2_info():
    mobilenet_v2 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model_info = get_model_info(mobilenet_v2, name='MobileNet-V2')
    print(model_info)

def get_MobileNetV3_info():
    mobilenet_v3 = MobileNetV3(mode='large')
    model_info = get_model_info(mobilenet_v3, name='MobileNetV3')
    print(model_info)

def get_ResNet50_info():
    resent50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
    model_info = get_model_info(resent50, 'ResNet-50')
    print(model_info)