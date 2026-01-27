import argparse

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from windtopocali.model.windtopocali import WindTopoCali
from windtopocali.dataset.dataset import WindTopoDataset
from windtopocali.utils.loss import CustomLoss
from windtopocali.utils.util import compute_metrics
parser = argparse.ArgumentParser()

parser.add_argument('--root_file',type=str,required=False,default='../data/WindTopoCali/')
parser.add_argument('--epochs',type=int,required=False,default=100)
parser.add_argument('--lr',type=float,required=False,default=5e-6)
parser.add_argument('--batch_size',type=int,required=False,default=32)
parser.add_argument('--device',type=str,required=False,default='cpu')

args = parser.parse_args()

root_file = args.root_file
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
device = args.device

if device == 'cuda' and torch.cuda.is_avaliable():
    device = 'cuda:0'
else:
    device = 'cpu'

loss_fn = CustomLoss()

weather_means = [0,0,0,0,0,0,0]
weather_std = [1,1,1,1,1,1,1]

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
topo_std = [1,1,1,1]

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

train_dataset = WindTopoDataset(
    root_file=root_file,
    train=True,
    weather_transform=weather_transform,
    weather_lr_transform=weather_lr_transform,
    topo_transform=topo_transform,
    topo_lr_transform=topo_lr_transform
)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

test_dataset = WindTopoDataset(
    root_file=root_file,
    train=True,
    weather_transform=weather_transform,
    weather_lr_transform=weather_lr_transform,
    topo_transform=topo_transform,
    topo_lr_transform=topo_lr_transform
)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=1)


model = WindTopoCali(
    n_weather_channels=7,
    wind_in_size=64,
    n_topo_channels=4,
    topo_in_size=128,
    device=device
)

optimizer = torch.optim.Adam(model.parameters(),betas=(0.9,0.999),lr=lr)

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        true_wind_spd = batch.pop('target')

        pred_wind_spd = model(batch)

        loss = loss_fn(pred_wind_spd,true_wind_spd)
        loss.backward()

        optimizer.step()

        mse = compute_metrics(pred_wind_spd,true_wind_spd)

        running_loss += loss.item()
        running_mse += mse.item()
        if i % 10 == 0:
            last_loss = running_loss / 10
            last_mse = running_mse / 10

            print('epoch {} | batch {} | loss {} | MSE {} '.format(epoch,i,last_loss,last_mse))

            running_loss = 0.
            running_mse = 0.



        

