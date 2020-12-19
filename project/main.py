import torch
import wandb
from time import perf_counter

import sys
sys.path.insert(0, './models')

from models.nasnet import NASNetALarge, NASNetAMobile
from models.mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from models.MnasNet import MnasNet
from models.mobilenet import MobileNetV3
from efficientnet_pytorch import EfficientNet
from models.DSNAS.load_DSNAS_model import load

def get_model_info(model, name):
    model.eval()
    model.to('cuda:0')

    run = wandb.init(project='sysdl-term-project', name=f'{name}-inference')
    wandb.watch(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters())
    
    inference_runs = 1000
    
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
        'GPU Name': torch.cuda.get_device_name(),
    }
    
    run.config.update(model_info)
    run.config.batch_size = input_tensor.shape[0]
    run.config.inference_runs = inference_runs
    run.finish()

    return model_info

model_infos = []

# try:
#     run = {'model': NASNetAMobile(), 'name': 'NASNetA-Small'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': NASNetALarge(), 'name': 'NASNetA-Large'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': fbnet('dmasking_f4'), 'name': 'FBNet-V2'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': MnasNet(), 'name': 'MNasNet'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False), 'name': 'Inception-V3'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': EfficientNet.from_name('efficientnet-b7'), 'name': 'EfficientNet-B7'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': EfficientNet.from_name('efficientnet-b0'), 'name': 'EfficientNet-B0'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=False), 'name': 'ResNeXt-101'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True), 'name': 'MobileNet-V2'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': MobileNetV3(mode='large'), 'name': 'MobileNet-V3'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False), 'name': 'ResNet-50'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': load(), 'name': 'DSNASsearch240'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

# try:
#     run = {'model': EfficientNet.from_name('efficientnet-b5'), 'name': 'EfficientNet-B5'}
#     model_infos.append(get_model_info(**run))
# except Exception as e:
#     print(e)
#     pass
# del run
# torch.cuda.empty_cache()

try:
    run = {'model': EfficientNet.from_name('efficientnet-b4'), 'name': 'EfficientNet-B4'}
    model_infos.append(get_model_info(**run))
except Exception as e:
    print(e)
    pass
del run
torch.cuda.empty_cache()

[print(f) for f in model_infos]