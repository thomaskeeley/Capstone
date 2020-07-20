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
    image_dir = '.../indo_test_image.tif',
    mask_dir = '.../indo_mask.geojson',
    out_dir = '.../desired/output/directory/',
    step = 4,
    prediction_threshold = 0.90,
    pos_chip_dir = ('.../positive_chips/'),
    neg_chip_dir = ('.../negative_chips/'),
    chip_width = 20,
    chip_height = 20,
    augment_pos_chips = True,
    augment_neg_chips = False,
    save_model = True,
    import_model = None)
    
    
    instance.execute()
    

#%%
