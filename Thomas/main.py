#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:12:13 2020

@author: thomaskeeley
"""

#%%

import warnings
warnings.filterwarnings('ignore')

#%%

from object_detection import ObjectDetection

#%%

if __name__ == '__main__':
    
    instance = ObjectDetection(
    working_dir = '...',
    import_image_path = '.../nk_aoi.tif',
    download_image = False,
    aoi_path = None,
    username = 'sirthomasnewton',
    password = 'Coronavirus1!',
    date_range = ('20190101', '20200101'),
    mask_path = '.../nk_aoi_b.geojson',
    out_dir = '...',
    step = 4,
    prediction_threshold = 0.90,
    pos_chip_dir = ('.../positive_chips/'),
    neg_chip_dir = ('.../negative_chips/'),
    chip_width = 20,
    chip_height = 20,
    augment_pos_chips = True,
    augment_neg_chips = False,
    save_model = False,
    import_model = None,
    segmentation = False)
    
    instance.execute()
    

#%%
