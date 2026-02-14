import numpy as np
import pandas as pd

import rasterio

import glob

from dataset.dataset import WindTopoDataset

weather_means = [0,0,0,0,0,0,0]
weather_std = [1,1,1,1,1,1,1]

topo_means = [0,0,0,0]
topo_std = [1,1,1,1]

root_file = './data/WindTopo/'

weather_file_list = glob.glob(root_file + 'train/*_rtma.tif')
topo_file_list = glob.glob(root_file + 'train/*_dem.tif')

weather_dset =[]
for file in weather_file_list:
    arr = rasterio.open(file).read()
    arr = arr.reshape(arr.shape[0],-1)
    weather_dset.append(arr)
weather_dset = np.concatenate(weather_dset,axis=1)
print(weather_dset.shape)

weather_channel_means = np.mean(weather_dset,axis=1)
weather_channel_stds = np.std(weather_dset,axis=1)

topo_dset = []
for file in topo_file_list:
    arr = rasterio.open(file).read()
    arr = arr.reshape(arr.shape[0],-1)
    topo_dset.append(arr)
topo_dset = np.concatenate(topo_dset,axis=1)
print(topo_dset.shape)

topo_channel_means = np.mean(topo_dset,axis=1)
topo_channel_stds = np.std(topo_dset,axis=1)

print(np.max(weather_dset,axis=1))
print(np.min(weather_dset,axis=1))
print(np.max(topo_dset,axis=1))
print(np.min(topo_dset,axis=1))


print(f'Weather Channel-wise Means: {weather_channel_means}')
print(f'Weather Channel-wise stdevs: {weather_channel_stds}')
print(f'Topography Channel-wise Means: {topo_channel_means}')
print(f'Topography Channel-wise stdevs: {topo_channel_stds}')