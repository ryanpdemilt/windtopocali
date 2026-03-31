import os
import argparse
import shutil
import tarfile
import glob

from pathlib import Path

import requests
from clint.textui import progress

import numpy as np
import pandas as pd
import geopandas as gpd

def process_asos_wind(gdf,qc_pass=['5']):
    wnd_str = gdf['wnd'].str.split(',',expand=True)
    dir_qa = wnd_str[1]
    wnd_fail_qa = (~dir_qa.isin(qc_pass))


    wnd_spd = wnd_str[3].astype(np.float64) / 10

    wnd_dir = wnd_str[0].astype(np.float64)

    wnd_u = wnd_spd * np.cos(np.deg2rad(wnd_dir))
    wnd_v = wnd_spd * np.cos(np.deg2rad(wnd_dir))

    wnd_spd[wnd_fail_qa] = -9999
    wnd_dir[wnd_fail_qa] = -9999
    wnd_u[wnd_fail_qa] = -9999
    wnd_v[wnd_fail_qa] = -9999

    gdf['wnd_spd'] = wnd_spd
    gdf['wnd_dir'] = wnd_dir
    gdf['uwnd'] = wnd_u
    gdf['vwnd'] = wnd_v

    return gdf



    

def process_asos_tmp(gdf,qc_pass=['5']):
    tmp_str = gdf['TMP'].str.split(',',expand=True)
    tmp_fail_qa = (~tmp_str[1].isin(qc_pass))

    tmp = tmp_str[0].astype(np.float64) / 10

    tmp[tmp_fail_qa] = -9999

    gdf['tmp'] = tmp

    return gdf

def process_asos_slp(gdf,qc_pass=['5']):
    slp_str = gdf['SLP'].str.split(',',expand=True)
    slp_fail_qa = (~slp_str[1].isin(qc_pass))

    atmp = slp_str[0].astype(np.float64) / 10
    atmp[slp_fail_qa] = -9999




def get_asos(dst,year,shp):

    isd_url = 'https://www.ncei.noaa.gov/pup/data/noaa/isd-history.csv'

    isd_dst = dst / Path('isd_history.csv')

    if not isd_dst.exists():
        print('=== Retrieving ISD Station History Document ===')
        r = requests.get(isd_url,stream=True)
        if r.status_code != 200:
                raise r.raise_for_status()
        
        with open(isd_dst,'wb') as out_file:
            shutil.copyfileobj(r.raw,out_file)

        print(f'ISD History Saved at: {isd_dst}')
    else:
        print(f'ISD History Document found at: {isd_dst}')

    isd_df = pd.DataFrame(isd_dst)
    isd_gdf = gpd.GeoDataFrame(isd_df,geometry=gpd.points_from_xy(isd_df.LON,isd_df.LAT),crs='EPSG:4326')

    tagged_stations = gpd.clip(isd_gdf,shp.to_crs('EPSG:4326'))
    tagged_stations['WBAN'] = tagged_stations['WBAN'].astype(str).str.zfill(5)
    tagged_stations['id'] = tagged_stations['USAF'] + tagged_stations['WBAN']

    url = f'https://www.ncei.noaa.gov/data/global-hourly/archive/csv/{year}.tar.gz'
    r = requests.get(url,stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()
    
    archive_dst = dst / Path(f'{year}.tar.gz')

    if not (archive_dst / Path(f'{year}')).is_dir():
        print(f'=== Downloading Global Hourly Archive for year: {year}')
        with open(archive_dst,'wb') as f:
            total_length = r.headers.get('content-length')
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()

        with tarfile.open(archive_dst) as f:
            f.extractall(path=archive_dst.parent)
    else:
        print(f'Weather Archive for year {year} found')

    station_file_list = glob.glob(archive_dst / Path(f'year') / '*.csv')
    filtered_file_list =[file for file in station_file_list if Path(file).parts[-1][:-4] in tagged_stations['id'].unique()]

    dfs = []
    for file in filtered_file_list:
        df = pd.read_csv(file,low_memory=False)
        dfs.append(df)

    full_df = pd.concat(dfs)

    full_df = process_asos_wind(full_df)
    full_df = process_asos_tmp(full_df)
    full_df = process_asos_slp(full_df)

    asos_attributes = ['STATION','DATE','LATITUDE','LONGITUDE','NAME','tmp','atmp','wnd_spd','wnd_dir','uwnd','vwnd']

    return full_df[asos_attributes]

    
    
def main(args):
    src = args.source
    year = args.year
    shp = Path(args.shp)
    dst = Path(args.dst)

    dst.mkdir(parents=True,exist_ok=True)

    shp_groups = gpd.read_file(shp)

    if src == 'asos':
        df = get_asos(dst,year,shp_groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--source',
        type=str,
        default='asos'
    )
    parser.add_argument(
        '--qa-flags',
        type=str,
        nargs='+',
        default=['5']
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2022
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        default='.\\data\\asos\\',
        help='Destination directory where weather files will be downloaded'
    )
    parser.add_argument(
        '--dst_name',
        type='str',
        default='output.csv'
    )
    parser.add_argument(
        '--shp',
        type=str,
        default='.\\data\\CalfireAdminZones',
        help='Shapefile to clip results to '
    )

    args = parser.parse_args()

    main(args)