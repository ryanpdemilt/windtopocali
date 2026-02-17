# windtopocali

## Introduction

High quality weather data is of immense value for fire behavior modeling, which both weather forecast models and weather station observations play a role in providing. Weather stations are an invaluable source of on the ground observations but are limited by their geographic sparsity and lack a means of extracting their insight beyond their local geographic area. Forecasting models are regularly gridded, standardized prediction models but are often too low resolution to meet the needs of fire behavior modeling on the ground and cannot adequately represent the complexities of the terrain which is heavily influential on the wind.

Machine learning models offer us a means to bridge this gap, offering models which can use computer-vision style modeling to derive insight on the association of gridded weather and topography data with observed weather station outputs. Using this we can re-plot an area using these outputs.

- Formal problem statement: Given a weather station, predict its hourly U,V weather components using the information of forecast data sourced from RTMA and topogrophy information in a 2.5 km radius around the station

## Table of Contents

## Dataset Curation

The datasets in this repository are sourced from the ASOS Integrated Surface Global Hourly Dataset, the Real Time Mesoscale Analysis forecasting model, and the SRTM Digital Elevation map. The necessary variables for each are listed here



The RTMA and SRTM datasets are reprojected to EPSG:5070 and retrieved from the Earth Engine Data Catalog. Stations are divided into training and testing points are collected hourly and then randomly sampled from available ASOS observations, only data which pass the quality checks and originate from an NCEI source are checked to ensure high quality observations. 

Each year has >3 million observations so data is not limited and needs to be sampled to an appropiate size. In the region of interest, there are ~300 operational stations.

- ASOS ISD
    - STATION Id
    - Station Latitude
    - Station Longitude
    - U-component of Wind
    - V-component of Wind
- RTMA @ 2.5km resolution (24x24 pixels)
    - HGT: model terrain elevation
    - TMP: temperature
    - SPFH: specific humidity
    - ACP01: total precipitation
    - GUST: wind gust speed
    - UGRD: u component of wind
    - VGRD: v component of wind
- SRTM @ 100m resampled resolution (400x400 pixels )
    - elevation
    - slope
    - aspect
    - mtpi



## Model Architecture + Pipeline

Weather Preprocess Chain:

                                (HR)
Raw RTMA Sample (24x24 pixels) ----> CenterCrop(16) -> Normalize -> Resize(64x64)
                              |
                              | (LR)
                               ----> CenterCrop(8)  -> Normalize -> Resize(64x64)
Topography Preprocess Chain

                                  (HR)
Raw SRTM Sample (400x400 pixels) ----> CenterCrop(400) -> Normalize -> Resize(128x128)
                                |
                                | (LR)
                                 ----> CenterCrop(100)  -> Normalize -> Resize(128x128)


Model Architecture:

- Input: LR & HR Weather, LR and HR Topography
- Ouptut: U & V components of Wind @ surface for a station centered on the region of the input images

1. 4 x Resnet 8 Feature Extractor (Weather_lr,weather_hr,topo_lr,topo_hr)
2. 2x Fusion Conv (HR & LR Topo+Weather)
3. FC Regression Block 4 layers 

- Choice of MSE loss or custom MSE based loss function with spread term
- CosineAnnealingLR Secheduler
- Adam Optimizer

## Model Evaluation

## Model Training

## Repository Layout

- model
    - windtopocali.py
- dataset
    - WindTopoDataset.py