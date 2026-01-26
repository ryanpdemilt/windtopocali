import argparse

import torch
import torchvision
from torchvision import transforms

from windtopocali.utils.loss import CustomLoss
parser = argparse.ArgumentParser()

parser.add_argument('--root_file',type=str,required=False,default='../data/WindTopoCali/')
parser.add_argument('--epochs',type=int,required=False,default=100)
parser.add_argument('--lr',type=float,required=False,default=5e-6)
parser.add_argument('--batch_size',type=int,required=False,default=128)

args = parser.parse_args()

optimizer = torch.optim.Adam(betas=(0.9,0.999))

loss = CustomLoss()

weather_means = [0,0,0,0,0,0,0]
weather_std = [0,0,0,0,0,0,0]

weather_transform = torch.nn.Sequential([
    transforms.CenterCrop(16),
    transforms.Normalize(mean=weather_means,std = weather_std),
    transforms.Resize(64)
])

weather_lr_transform = torch.nn.Sequential([
    transforms.CenterCrop(8),
    
    transforms.Resize(64)
])

topo_means = [0,0,0,0]
topo_std = [0,0,0,0]

topo_transform = torch.nn.Sequential([
    transforms.CenterCrop(400),
    transforms.Normalize(mean = topo_means,std=topo_std),
    transforms.Resize(128)
])

topo_lr_transform = torch.nn.Sequential([
    transforms.CenterCrop(100),
    transforms.Normalize(mean = topo_means,std=topo_std),
    transforms.Resize(128)
])

