import requests

import io
import itertools
from retry import retry
import multiprocessing as mp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ee

import torch
import torchvision
from torchvision import transforms

import shapely.geometry
from shapely.geometry import Point

import geopandas as gpd

from model.windtopocali import WindTopoCali

ee_crs = 'EPSG:4326'
meter_crs = 'EPSG:5070'
gsd = 100
rtma_gsd = 2500

weather_means = [0,0,0,0,0,0,0]
weather_std = [1,1,1,1,1,1,1]
topo_means = [0,0,0,0]
topo_std = [1,1,1,1]

ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com',project='eofm-benchmark')

center_lon, center_lat = -123.14, 41.12
time = '2024-01-01T00:00:00'
roi_size = 200

point = Point([center_lon,center_lat])

pp = gpd.GeoSeries(point).set_crs(ee_crs).to_crs(meter_crs).buffer(roi_size)

top, left, bottom, right = pp.total_bounds

gdf_grid = gpd.GeoDataFrame(
    geometry=[
        shapely.geometry.Point(x,y)
        for x in np.arange(top,bottom,gsd)
        for y in np.arange(left,right,gsd)
    ],
    crs=meter_crs
).to_crs(ee_crs)

dt = ee.Date(time)

rtma_proj = ee.Projection(crs='EPSG:5070').atScale(2500)
rtma_projection = rtma_proj.getInfo()

topo_proj = ee.Projection(crs='EPSG:5070').atScale(100)
topo_projection = topo_proj.getInfo()

# @retry(tries=100,delay=3,backoff=2)
def build_request(idx,point,date=dt,proj_rtma=rtma_projection,proj_topo=topo_projection):
    lon, lat = point.x, point.y

    rtma_proj = ee.Projection(crs='EPSG:5070').atScale(2500)
    rtma_projection = rtma_proj.getInfo()

    topo_proj = ee.Projection(crs='EPSG:5070').atScale(100)
    topo_projection = topo_proj.getInfo()

    print(lon,lat)
    print(date.getInfo())

    rtma_point = ee.Geometry.Point(coords=[lon,lat]).transform(rtma_proj)
    rtma_box = rtma_point.buffer(distance=25000).bounds(proj=rtma_proj,maxError=1)

    topo_point = ee.Geometry.Point(coords=[lon,lat]).transform(topo_proj)
    topo_box = topo_point.buffer(distance=25000).bounds(proj=topo_proj,maxError=1)


    #terrain
    dem = ee.Image('USGS/SRTMGL1_003').clip(topo_box).select('elevation')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    mtpi = ee.Image('CSP/ERGo/1_0/Global/SRTM_mTPI').clip(topo_box).select('elevation').rename('mtpi')

    topo_frame = dem.addBands(slope).addBands(aspect).addBands(mtpi)


    topo_crs = proj_topo['crs']
    topo_trns = proj_topo['transform']

    topo_url = topo_frame.getDownloadURL(
        {
            'bands':['elevation','slope','aspect','mtpi'],
            'region':topo_box,
            'scale':100,
            'crs':topo_crs,
            'crsTransform':topo_trns,
            'format':'NPY'
        }
    )
    
    # print(f'Topo Download URL: {topo_url}')

    #weather
    rtma_frame = ee.ImageCollection('NOAA/NWS/RTMA') \
        .filterDate(date,date.advance(1,'hour')) \
        .filterBounds(rtma_box) \
        .first() \
        .clip(rtma_box) \
        .select(['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'])
    
    rtma_crs = proj_rtma['crs']
    rtma_transform = proj_rtma['transform']

    rtma_url = rtma_frame.getDownloadURL(
        {
            'bands':['TMP','UGRD','VGRD','SPFH','ACPC01','HGT','GUST'],
            'region':rtma_box,
            'scale':2500,
            'crs':rtma_crs,
            'crsTransform':rtma_transform,
            'format':'NPY'
        }
    )

    print(f'RTMA Download URL {rtma_url}')

    response = requests.get(rtma_url)
    if response.status_code != 200:
        raise response.raise_for_status()
    rtma_data = np.load(io.BytesIO(response.content))
    
    print(f'ID: {idx} RTMA loaded')
    
    response = requests.get(topo_url)
    if response.status_code != 200:
        raise response.raise_for_status()

    topo_data = np.load(io.BytesIO(response.content))
    print(f'ID: {idx} Topo Loaded')
    

    data = {
        'coordinates':[lon,lat],
        'rtma':rtma_data,
        'topo':topo_data
    }

    return idx,data

test_idx, test_data = build_request(idx=0,point=gdf_grid.geometry[0],date=dt,proj_rtma=rtma_projection,proj_topo=topo_projection)
