from torch.utils.data import Dataset
import torch

import os
import glob
import datetime

import numpy as np
import pandas as pd

import tifffile

class WindTopoDataset(Dataset):
    def __init__(self,
                 root_file='../data/WindTopoCali/',
                 target_file='norcal_wind_data.csv',
                 train=True,
                 weather_transform=None,
                 weather_lr_transform = None,
                 topo_transform=None,
                 topo_lr_transform = None,
                 climate_vars = ['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'],
                 topo_vars = ['elevation','slope','aspect','mtpi']
    ):        
        self.root_file = root_file
        self.target_file = target_file
        self.train = train

        self.weather_transform = weather_transform
        self.weather_lr_transform = weather_lr_transform

        self.topo_transform = topo_transform
        self.topo_lr_transform = topo_lr_transform

        self.climate_vars = climate_vars
        self.topo_vars = topo_vars

        if self.train:
            data_file = root_file + 'train/'
        else:
            data_file = root_file + 'test/'

        self.file_list = glob.glob(data_file + '*.tif')

        self.rtma_files = [file for file in self.file_list if 'rtma' in file]
        self.topo_files = [file for file in self.file_list if 'dem' in file]

        self.topo_lookup = dict(zip([file.split('_')[0].split('/')[-1] for file in self.topo_files],[tifffile.imread(file) for file in self.topo_files]))

        self.station_data_fname = self.root_file + self.target_file
        self.station_data = pd.read_csv(self.station_data_fname)
        self.station_data['DATE'] = pd.to_datetime(self.station_data['DATE'],format='%Y-%m-%dT%H:%M:%S')
        self.station_data['DATE'] = self.station_data['DATE'].dt.round('h')

        self.station_data = self.station_data.drop_duplicates(['STATION','DATE'])
        self.targets_list = []
        self.matching_topos = []
        for rtma_sample_fname in self.rtma_files:
            station, year,month,day,hour, _= rtma_sample_fname.split('_')
            station = station.split('/')[-1]

            dt = datetime.datetime(int(year),int(month),int(day),int(hour))

            target_row = self.station_data[(self.station_data['STATION'] == station) & (self.station_data['DATE'] == dt)]
            uwnd = target_row['UWND'].item()
            vwnd = target_row['VWND'].item()

            matching_topo = self.topo_lookup[station]

            self.targets_list.append([uwnd,vwnd])
            self.matching_topos.append(matching_topo)


    def __len__(self):
        return len(self.rtma_files)
    def __getitem__(self,idx):
        rtma_sample_fname = self.rtma_files[idx]

        station, year,month,day,hour, _= rtma_sample_fname.split('_')
        station = station.split('/')[-1]

        dt = datetime.datetime(int(year),int(month),int(day),int(hour))

        rtma_tensor = torch.from_numpy(tifffile.imread(rtma_sample_fname).transpose((2,0,1)).astype(np.float32))

        if self.weather_transform is not None:
            rtma_hr_tensor = self.weather_transform(rtma_tensor)
        else:
            rtma_hr_tensor = rtma_tensor

        if self.weather_lr_transform is not None:
            rtma_lr_tensor = self.weather_lr_transform(rtma_tensor)

        topo_tensor = torch.from_numpy(self.matching_topos[idx].copy().transpose((2,0,1)).astype(np.float32))

        if self.topo_transform is not None:
            topo_hr_tensor = self.topo_transform(topo_tensor)
        else:
            topo_hr_tensor = topo_tensor

        if self.topo_lr_transform is not None:
            topo_lr_tensor = self.topo_lr_transform(topo_tensor)
        else:
            topo_lr_tensor = topo_tensor

        rtma_hr_tensor = rtma_hr_tensor.float()
        rtma_lr_tensor = rtma_lr_tensor.float()

        topo_hr_tensor = topo_hr_tensor.float()
        topo_lr_tensor = topo_lr_tensor.float()

        uwnd, vwnd = self.targets_list[idx]

        target = torch.tensor([uwnd,vwnd]).float()

        output = {
            'rtma':rtma_hr_tensor,
            'rtma_lr':rtma_lr_tensor,
            'topo':topo_hr_tensor,
            'topo_lr':topo_lr_tensor,
            'target':target
        }

        return output



    