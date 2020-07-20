#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:50:53 2020

@author: thomaskeeley
"""


from sentinelsat import SentinelAPI
import geopandas as gpd

#%%

user = 'username' 
password = 'password' 
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

#%%

shape = gpd.read_file('.../aoi_polygon.geojson')

footprint = None
for i in shape['geometry']:
    footprint = i


products = api.query(footprint,
                     date = ('20190601', '20190615'),
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-2A',
                     cloudcoverpercentage = (0,10)
                    )



products_gdf = api.to_geodataframe(products)
products_gdf_sorted = products_gdf.sort_values(['cloudcoverpercentage'], ascending=[True])
products_gdf_sorted

#%%

key = products_gdf_sorted.index[0]

api.download(key)

