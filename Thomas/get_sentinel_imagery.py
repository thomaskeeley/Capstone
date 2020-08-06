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
        
        self.image_dir = image_dir
        self.aoi_dir = aoi_dir
        self.username = username
        self.password = password
        self.date_range = date_range
        
    def get_image_products(self):
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
    

    # def clip_image(self):
    #     image_path = self.download_image()
    #     image = rasterio.open(image_path)
    #     crs = str(image.crs).split(':')[1] 
    #     shape = gpd.read_file(self.aoi_dir)
    #     shape = shape.to_crs({'init': 'epsg:{}'.format(crs)})
    #     print('\n >>> CROPPING IMAGE')
    #     with fiona.open(self.aoi_dir, "r") as shape:
    #             geoms = [feature["geometry"] for feature in shape]
            
    #             clipped_image, out_transform = mask(image, geoms, crop=True)
            
    #     im_tensor = clipped_image.transpose(1, 2, 0)
    #     plt.figure(figsize = (15, 30))
    #     plt.subplot(3, 1, 1)
    #     plt.imshow(im_tensor)
        
    #     return im_tensor
    

#%%

# image_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/'
# aoi_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/test_aoi_small.geojson'
# username = 'sirthomasnewton'
# password = 'Coronavirus1!'
# date_range = ('20190101', '20200101')
        
# instance = GetSentinelImagery(image_dir, aoi_dir, username, password, date_range)
        
# #%%

# image = instance.clip_image()

# #%%

# with rasterio.Env():

#     # Write an array as a raster band to a new 8-bit file. For
#     # the new file's profile, we start with the profile of the source
#     profile = clipped_image.profile

#     # And then change the band count to 1, set the
#     # dtype to uint8, and specify LZW compression.
#     profile.update(
#         dtype=rasterio.uint8,
#         count=1,
#         compress='lzw')

#     with rasterio.open('example.tif', 'w', **profile) as dst:
#         dst.write(clipped_image.astype(rasterio.uint8), 1)






# im = Image.fromarray(clipped_image)
# im.save('test.tif')

# out_transform_copy = out_transform.copy()
# with fiona.open("clipped_image.tif", "w", **out_transform) as final:
#     final.write(clipped_image)

# image.bands
# im_bands = im.count
        
#         im_width = im.width
#         im_height = im.height
        
#         left, bottom, right, top = im.bounds
        
#         im_x_res = (right-left)/im_width
#         im_y_res = (top-bottom)/im_height  
        
#         crs = str(im.crs).split(':')[1] 