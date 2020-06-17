#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:25:44 2020

@author: thomaskeeley
"""

from PIL import Image
import glob
import os

#%%

# Define the desired dimension of image chips
out_width = 80
out_height = 80

#%%

# Change directory to first folder of positive chips and get filenames
os.chdir(".../image_chips_60m/")

pos_filenames = [i for i in glob.iglob('*.tif')]

# Reach in each image crop edges to match desired dimension, transform/replicate to create additional chips
pos_chips = []
for file in pos_filenames:
    
    im = Image.open(file)
    
    width, height = im.size
    
    width_min = round((width - out_width) / 2)
    width_max = width_min + out_width
    
    height_min = round((height - out_height) / 2)
    height_max = height_min + out_width
    
    im = im.crop((width_min, height_min, width_max, width_max))
    
    pos_chips.append(im)
    
    pos_chips.append(im.transpose(Image.FLIP_LEFT_RIGHT))
    
    pos_chips.append(im.transpose(Image.FLIP_TOP_BOTTOM))
    
    pos_chips.append(im.rotate(90))
    
    pos_chips.append(im.rotate(180))
    
#%%

# Change directory to second folder of positive chips and get filenames
os.chdir(".../image_chips_60m_2/")

pos_filenames2 = [i for i in glob.iglob('*.tif')]

# Reach in each image crop edges to match desired dimension, transform/replicate to create additional chips
for file in pos_filenames2:
    
    im = Image.open(file)
    
    width, height = im.size
    
    width_min = round((width - out_width) / 2)
    width_max = width_min + out_width
    
    height_min = round((height - out_height) / 2)
    height_max = height_min + out_width
    
    im = im.crop((width_min, height_min, width_max, width_max))
    
    pos_chips.append(im)
    
    pos_chips.append(im.transpose(Image.FLIP_LEFT_RIGHT))
    
    pos_chips.append(im.transpose(Image.FLIP_TOP_BOTTOM))
    
    pos_chips.append(im.rotate(90))
    
    pos_chips.append(im.rotate(180))

#%%

# Change directory to folder of negative chips and get filenames
os.chdir(".../image_chips_60m_random/")

neg_filenames = [i for i in glob.iglob('*.tif')]

# Reach in each image crop edges to match desired dimension, transform/replicate to create additional chips
neg_chips = []
for file in neg_filenames:
    
    im = Image.open(file)
    
    width, height = im.size
    
    width_min = round((width - out_width) / 2)
    width_max = width_min + out_width
    
    height_min = round((height - out_height) / 2)
    height_max = height_min + out_width
    
    im = im.crop((width_min, height_min, width_max, width_max))
    
    neg_chips.append(im)

    neg_chips.append(im.transpose(Image.FLIP_LEFT_RIGHT))
    
    neg_chips.append(im.transpose(Image.FLIP_TOP_BOTTOM))
    
    neg_chips.append(im.rotate(90))
    
    neg_chips.append(im.rotate(180))

#%%

# Change directory to desired output folder and save image chips
os.chdir(".../chips/")


for i, im in enumerate(pos_chips):
    im.save(f"chip_{i}_pos.png")


neg_idx = len(pos_chips)

for i, im in enumerate(neg_chips):
    idx = i + neg_idx
    im.save(f"chip_{idx}_neg.png")
    
    
#%%


