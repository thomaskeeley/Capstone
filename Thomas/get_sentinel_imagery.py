#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:48:02 2020

@author: thomaskeeley
"""

import os
from zipfile import ZipFile
from sentinelsat import SentinelAPI
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
from PIL import Image

import matplotlib.pyplot as plt

#%%

class GetSentinelImagery:
    
    def __init__(self, image_dir, aoi_dir, username, password, date_range):
        """
        

        Parameters
        ----------
        image_dir : Directory
            Desired output path of downloaded imagery
        aoi_dir : Directory
            Filepath to AOI file to be used in querying data
        username : String
            Copernicus Hub username
        password : String
            Copernicus Hub password
        date_range : Tuple
            Desired start and end date for querying data

        Returns
        -------
        None.

        """
        self.image_dir = image_dir
        self.aoi_dir = aoi_dir
        self.username = username
        self.password = password
        self.date_range = date_range
        
    def get_image_products(self):
        """
        

        Returns
        -------
        products : List
            Contains collection of information of available images from query

        """
        print('\n >>> CONDUCTING API QUERY')
        api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')
        shape = gpd.read_file(self.aoi_dir)
        shape = shape.to_crs({'init': 'epsg:4326'})

        footprint = None
        for i in shape['geometry']:
            footprint = i
            
        products = api.query(footprint,
                             date = self.date_range,
                             platformname = 'Sentinel-2',
                             processinglevel = 'Level-2A',
                             cloudcoverpercentage = (0,10)
                             )
        
        products = api.to_geodataframe(products)
        products = products.sort_values(['cloudcoverpercentage'], ascending=[True])
        
        return products
    
    
    def download_image(self):
        """
        

        Returns
        -------
        image_path : Directory
            Filepath to downloaded image after unzipping
        date : String
            Date of downloaded image
        time : String
            Zulu time of downloaded image

        """
        api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')
        products = self.get_image_products()
        key = products.index[0]
        title = products['title'][0]
        date, time = str(products['endposition'][0]).split(' ')
        os.chdir(self.image_dir)
        print('\n >>> DOWNLOADING IMAGE')
        api.download(key)
        
        with ZipFile(title + '.zip', 'r') as zipObject:
            filenames = zipObject.namelist()
            for file in filenames:
                if file.endswith('TCI_10m.jp2'):
                    image_path = zipObject.extract(file)
        os.remove(title + '.zip')           
        return image_path, date, time
    
