from torch.utils.data import Dataset
import torch

import os
import glob
import datetime

import numpy as np
import pandas as pd

import rasterio

class WindTopoDataset(Dataset):
    def __init__(self,
                 root_file='../data/WindTopoCali/',
                 train=True,
                 weather_transform=None,
                 topo_transform=None,
                 climate_vars = ['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'],
                 topo_vars = ['elevation','slope','aspect','mtpi']
    ):
        
        self.root_file = root_file
        self.train = train
        self.weather_transform = weather_transform
        self.topo_transform = topo_transform

        self.climate_vars = climate_vars
        self.topo_vars = topo_vars

        if self.train:
            data_file = root_file + 'train/'
        else:
            data_file = root_file + 'test/'

        self.file_list = glob.glob(data_file + '*.tif')

        self.rtma_files = [file for file in self.file_list if 'rtma' in file]
        self.topo_files = [file for file in self.file_list if 'dem' in file]

        self.topo_lookup = dict(zip([file.split('_')[0] for file in self.topo_files],self.topo_files))

        self.station_data_fname = self.data_root + 'norcal_wind_data.csv'
        self.station_data = pd.read_csv(self.station_data_fname) 


    def __len__(self):
        return len(self.rtma_files)
    def __getitem__(self,idx):
        rtma_sample_fname = self.rtma_files[idx]

        station, year,month,day,hour, _= rtma_sample_fname.split('_')

        dt = datetime.datetime(year,month,day,hour)

        topo_file_fname = self.topo_lookup[station]

        rtma_src = rasterio.open(rtma_sample_fname)
        rtma_arr = rtma_src.read()

        rtma_tensor = torch.from_numpy(rtma_arr)

        if self.weather_transform is not None:
            rtma_tensor = self.weather_transform(rtma_tensor)

        topo_src = rasterio.open(topo_file_fname)
        topo_arr = topo_src.read()

        if self.topo_transform is not None:
            topo_tensor = self.topo_transform(topo_tensor)

        topo_tensor = torch.from_numpy(topo_arr)

        target_row = self.station_data[self.station_data['STATION'] == station and self.station_data['DATE'] == dt]
        target = torch.tensor([target_row['UWND','VWND']])

        output = {
            'rtma':rtma_tensor,
            'topo':topo_tensor,
            'target':target,
            'date':dt
        }

        return output



    