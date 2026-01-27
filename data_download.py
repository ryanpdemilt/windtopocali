import ee
import os
import glob
import argparse

import retry
from retry import retry
import requests

import multiprocessing
import shutil

import calendar
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--sample_file',type=str,default='sample.csv',required=False)
parser.add_argument('--output_folder',type=str,default='./data/WindTopoCali/',required=False)

ee.Authenticate()
ee.Initialize(
    opt_url='https://earthengine-highvolume.googleapis.com',
    project='eofm-benchmark'
)
sample_file = '/content/drive/My Drive/WindTopoCaliFull/sample_2021.csv'

samples = pd.read_csv(sample_file)
samples['DATE'] = pd.to_datetime(samples['DATE'],format='%Y-%m-%dT%H:%M:%S')
samples['DATE'] = samples['DATE'].dt.round('h')

rtma_scale = 2500
srtm_scale = 30
srtm_reduced_scale = 100

crs = 'EPSG:5070'
meter_proj = ee.Projection(crs=crs).atScale(100)
meter_projection = meter_proj.getInfo()

rtma_proj = ee.Projection(crs=crs).atScale(2500)
rtma_projection = rtma_proj.getInfo()

def get_elevation(geometry):
  dem = ee.Image('USGS/SRTMGL1_003') \
    .clip(geometry) \
    .select('elevation')

  slope = ee.Terrain.slope(dem).rename('slope')
  aspect = ee.Terrain.aspect(dem).rename('aspect')

  mtpi = ee.Image('CSP/ERGo/1_0/Global/SRTM_mTPI') \
    .clip(geometry) \
    .select('elevation').rename('mtpi')
  return dem, slope, aspect, mtpi

def get_rtma_tile(geometry,year,month,day,hour):
  rtma = ee.ImageCollection('NOAA/NWS/RTMA')
  time_code = f'{year}-{(str(month)).zfill(2)}-{(str(day)).zfill(2)}T{(str(hour)).zfill(2)}:00:00'
  start_time = ee.Date(time_code)
  end_time = start_time.advance(1,'hour')

  weather_sample = rtma.filter(ee.Filter.date(start_time,end_time)) \
                        .filterBounds(geometry) \
                        .first() \
                        .clip(geometry) \
                        .select(['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'])

  return weather_sample

@retry(tries=10,delay=1,backoff=2)
def get_station_dem_request(station):
  record = samples[samples['STATION'] == station].iloc[0]
  station = record['STATION']
  lat, lon = record['LATITUDE'], record['LONGITUDE']

  meter_point = ee.Geometry.Point([lon,lat]).transform(meter_proj)
  meter_box = meter_point.buffer(distance=25000).bounds(proj=meter_proj,maxError=0.1)

  dem_sample, slope_sample, aspect_sample, mtpi_sample = get_elevation(meter_box)
  dem_sample_combine = dem_sample.addBands(slope_sample).addBands(aspect_sample).addBands(mtpi_sample)

  fname = f'{station}'

  dem_fname = '/content/drive/My Drive/WindTopoCaliFull/train/' + fname + '_dem.tif'

  url = dem_sample_combine.getDownloadURL(
      {
          'bands':['elevation','slope','aspect','mtpi'],
          'region':meter_box,
          'crs':meter_projection['crs'],
          'crsTransform':meter_projection['transform'],
          'scale':100,
          'format':'GEO_TIFF'
      }
  )
  r = requests.get(url,stream=True)
  if r.status_code != 200:
    raise r.raise_for_status()

  with open(dem_fname,'wb') as out_file:
    shutil.copyfileobj(r.raw,out_file)
  print('Done:', station)

@retry(tries=100,delay=3,backoff=2)
def get_rtma_sample_request(record):
  lat, lon = record['LATITUDE'], record['LONGITUDE']
  date = record['DATE']
  station = record['STATION']

  rtma_point = ee.Geometry.Point([lon,lat]).transform(rtma_proj)
  rtma_box = rtma_point.buffer(distance=25000).bounds(proj=rtma_proj,maxError=0.1)

  year, month, day, hour = date.year, date.month, date.day, date.hour

  rtma_sample = get_rtma_tile(rtma_box,year,month,day,hour)

  fname = f'{station}_{year}_{month}_{day}_{hour}'


  rtma_fname = '/content/drive/My Drive/WindTopoCaliFull/train' + fname + '_rtma.tif'

  url = rtma_sample.getDownloadURL(
      {
          'bands':['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'],
          'region':rtma_box,
          'scale':2500,
          'crs':rtma_projection['crs'],
          'crsTransform':rtma_projection['transform'],
          'format':'GEO_TIFF'
      }
  )
  r = requests.get(url,stream=True)
  if r.status_code != 200:
    raise r.raise_for_status()

  with open(rtma_fname,'wb') as out_file:
    shutil.copyfileobj(r.raw,out_file)
  print('Done:', station)


unique_stations = pd.unique(samples['STATION'])

pool = multiprocessing.Pool(25)
pool.map(get_station_dem_request,list(unique_stations))
pool.close()
pool.join()

pool = multiprocessing.Pool(25)

pool.map(get_rtma_sample_request,samples.to_dict('records'))
pool.close()
pool.join()