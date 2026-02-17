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
- RTMA @ 2.5km resolution
    - HGT: model terrain elevation
    - TMP: temperature
    - SPFH: specific humidity
    - ACP01: total precipitation
    - GUST: wind gust speed
    - UGRD: u component of wind
    - VGRD: v component of wind
- SRTM @ 100m resampled resolution
    - elevation
    - slope
    - aspect
    - mtpi

## Model Architecture



## Model Evaluation



## Model Training

## Repository Layout