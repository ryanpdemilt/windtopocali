import os
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

import torchvision
from torchvision import transforms

from model.windtopocali import WindTopoCali
from dataset import WindTopoDataset
from utils.loss import CustomLoss
from utils.util import compute_metrics

from torch.profiler import profile,record_function,ProfilerActivity

parser = argparse.ArgumentParser()

parser.add_argument('--root_file',type=str,required=False,default='../data/WindTopoCali/')
parser.add_argument('--target_file',type=str,required=False,default='sample.csv')
parser.add_argument('--epochs',type=int,required=False,default=100)
parser.add_argument('--lr',type=float,required=False,default=1e-3)
parser.add_argument('--batch_size',type=int,required=False,default=32)
parser.add_argument('--device',type=str,required=False,default='cpu')
parser.add_argument('--log_file',type=str,required=False,default='log.txt')
parser.add_argument('--model_save',type=str,required=False,default='checkpoints/')
parser.add_argument('--exp_name',type=str,required=False,default='test')
parser.add_argument('--loss_name',type=str,required=False,default='custom')
args = parser.parse_args()

root_file = args.root_file
target_file = args.target_file
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
device = args.device
log_file = args.log_file
exp_name = args.exp_name
model_save = args.model_save
loss_name = args.loss_name

out_dir = os.path.join(model_save,exp_name)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
model_save = out_dir


if device == 'cuda' and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if loss_name == 'mse':
    loss_fn = torch.nn.MSELoss()
elif loss_name == 'custom':
    loss_fn = CustomLoss()

weather_means = [ 1.56e1, 1.40, -1.55e-1, 6.57e-3, 6.30e-2, 4.49e2, 6.12]
weather_std = [8.21, 2.95, 3.09, 2.61e-3, 4.07e-1, 6.09e2, 3.83]

weather_transform = torch.nn.Sequential(
    transforms.CenterCrop(16),
    transforms.Normalize(mean=weather_means,std = weather_std),
    transforms.Resize(64)
)

weather_lr_transform = torch.nn.Sequential(
    transforms.CenterCrop(8),
    transforms.Normalize(mean=weather_means,std = weather_std),
    transforms.Resize(64)
)

topo_means = [4.02e2, 6.1135697e+00,  1.46e2, -2.69e8]
topo_std = [6.89e2, 7.82, 1.13e2, 7.11e8]

topo_transform = torch.nn.Sequential(
    transforms.CenterCrop(400),
    transforms.Normalize(mean = topo_means,std=topo_std),
    transforms.Resize(128)
)

topo_lr_transform = torch.nn.Sequential(
    transforms.CenterCrop(100),
    transforms.Normalize(mean = topo_means,std=topo_std),
    transforms.Resize(128)
)

train_dataset = WindTopoDataset(
    root_file=root_file,
    target_file=target_file,
    train=True,
    weather_transform=weather_transform,
    weather_lr_transform=weather_lr_transform,
    topo_transform=topo_transform,
    topo_lr_transform=topo_lr_transform
)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          persistent_workers=True,
                          prefetch_factor=4
                        #   pin_memory=True
                    )


num_batches = len(train_loader)

test_dataset = WindTopoDataset(
    root_file=root_file,
    target_file=target_file,
    train=False,
    weather_transform=weather_transform,
    weather_lr_transform=weather_lr_transform,
    topo_transform=topo_transform,
    topo_lr_transform=topo_lr_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)



model = WindTopoCali(
    n_weather_channels=7,
    wind_in_size=64,
    n_topo_channels=4,
    topo_in_size=128,
    device=device
)
print(f'Model Parameters: {sum(p.numel() for p in model.parameters())}')
print(f'saving checkpoints in: {model_save}')

running_loss = 0.
running_mse = 0.

if device != 'cpu':
    model = model.to(device)
    loss = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(),betas=(0.9,0.999),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.95)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=20)

# with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,acc_events=True) as prof:
#     with record_function('model_inference'):
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        true_wind_spd = batch.pop('target')

        if device != 'cpu':
            true_wind_spd = true_wind_spd.to(device)
            for x in batch.keys():
                batch[x] = batch[x].to(device)        

        pred_wind_spd = model(batch)

        loss = loss_fn(pred_wind_spd,true_wind_spd)
        loss.backward()

        optimizer.step()
        
        mse = compute_metrics(true_wind_spd,pred_wind_spd)

        running_loss += loss.item()
        running_mse += mse.item()
        if i % 10 == 0:
            last_loss = running_loss / 10
            last_mse = running_mse / 10

            rmse = np.sqrt(last_mse)

            test_loss = 0.0
            test_metrics = []
            

            with torch.no_grad():
                for j,test_batch in enumerate(test_loader): 
                    true_wind_spd = test_batch.pop('target')
                    if device != 'cpu':
                        true_wind_spd = true_wind_spd.to(device)
                        for x in test_batch.keys():
                            test_batch[x] = test_batch[x].to(device)
                    pred_wind_spd = model(test_batch)
                    test_loss = test_loss + loss_fn(pred_wind_spd,true_wind_spd).item()
                    test_metrics.append(compute_metrics(true_wind_spd,pred_wind_spd).item())
            test_rmse = np.sqrt(np.mean(test_metrics))
            test_loss = test_loss / (j+1)

            log_str = 'epoch {} | batch {} / {} | loss {} | MSE {} | RMSE {} | test-loss {} | test-rmse {}'.format(epoch,i,num_batches,last_loss,last_mse,rmse,test_loss,test_rmse)
            

            print(log_str)

            with open(log_file,'a') as f:
                f.write(log_str + '\n')

            running_loss = 0.
            running_mse = 0.

    scheduler.step()
    if epoch % 10 == 0:
        model_ckpt_save = os.path.join(model_save,f'epoch_{epoch}_checkpoint.pth')
        torch.save(model.state_dict(),model_ckpt_save)

# print(prof.key_averages().table(sort_by='cpu_time_total',row_limit=10))

outpath = os.path.join(model_save,'final_pretrained_model.pth')
torch.save(model.state_dict(),outpath)



        

