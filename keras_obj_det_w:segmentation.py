#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:45:49 2020

@author: thomaskeeley
"""


#%%
import numpy as np
import pandas as pd
import os
import sys
import glob

import cv2

import datetime

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from shapely.geometry import Polygon

import rasterio
import rasterio.features

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from PIL import Image

from matplotlib import pyplot as plt


#%%

# Change directory to folder containing training chips, read in file names and put in order by ID
os.chdir(".../chips/")

chip_filenames = [i for i in glob.iglob('*.png')]

chip_filenames = sorted(chip_filenames, key=lambda x: int("".join([i for i in x if i.isdigit()])))

# Read in chips as arrays
chips = []
for file in chip_filenames:
    
    im = Image.open(file)
    imarray = np.array(im).astype('uint8')
    imarray = np.moveaxis(imarray, -1, 0)
    chips.append(imarray)
    

chips = np.array(chips).astype('uint8')

# Create training labels based on pos/neg tag in filename
chip_labels = np.zeros(len(chip_filenames)).astype('float32')
for index, item in enumerate(chip_filenames):
    if 'pos' in item:
        chip_labels[index] = 1


chip_labels = np_utils.to_categorical(chip_labels, 2)


#%%

X = chips

y = chip_labels

# shuffle all indexes
indexes = np.arange(len(X))
np.random.shuffle(indexes)

X = X[indexes].transpose([0,2,3,1])
y = y[indexes]

# normalization
X = X / 255


#%%

# Define dimension of training chips, define X and combine training data
n_spectrum = 3 # color chanel (RGB)
width = 80
height = 80


#%%

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


#%%

# Define the model and compile
model = Sequential([
    Conv2D(32, (3,3),  activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Flatten(),
    Dense(400, activation="tanh"),
    Dropout(0.25),
    BatchNormalization(),
    Dense(2, activation="softmax")
])

model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])


#%%

model.fit(X_train, y_train, batch_size=32, epochs=18, validation_split=0.2, shuffle=True, verbose=2)

# Train model
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), shuffle=True)


#%%

# Evaluate model
model.evaluate(X_test, y_test, verbose=2)


#%%

# Define filepath for image to be used in object detection
filepath = '.../test_aoi_clip2_water.tif'

# Read in the image and define image parameters  
    
im = rasterio.open(filepath)   

im_bands = im.count

im_width = im.width
im_height = im.height

left, bottom, right, top = im.bounds

im_x_res = (right-left)/im_width
im_y_res = (top-bottom)/im_height  

crs = str(im.crs).split(':')[1]

# Create tensor of image data
im_tensor = im.read()
im_tensor = im_tensor.transpose(1, 2, 0)

#%%

# Plot/View image
plt.figure(1, figsize = (15, 30))

plt.subplot(3, 1, 1)
plt.imshow(im_tensor)

plt.show()

#%%

# Conduct K Means Classification Image Segmentation

original_image = cv2.imread('.../test_aoi_clip2_water.tif')

img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

# img=cv2.cvtColor(im_tensor,cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

#%%

# Display results of image segmentation

plt.figure(1, figsize = (15, 30))

plt.subplot(3, 1, 1)
plt.legend()
plt.imshow(result_image)


plt.show()

#%%

# Get the classes and pixel counts for each

unique, counts = np.unique(result_image, return_counts=True)

#%%

# Create list of box coordinates around each minority class pixel
# These locations will be used to run predictions based on the assumption that the 
# minority class in the K Means segmentation is the target of object detection

y_shape = result_image.shape[0]
x_shape = result_image.shape[1]
test_areas = []
target = unique[-3]
segment_pix = np.where(result_image == target)
for i in range(0, len(segment_pix[0])):
    
    row = segment_pix[0][i]
    col = segment_pix[1][i]
    
    if row <= 40:
        y_min = 0
        y_max = 80
    if row >= 40:
        y_min = row - 40
        if row + 40 >= y_shape:
            y_max = y_shape
            y_min = y_max - 80
        else: 
            y_max = row + 40
        
    if col <= 40:
        x_min = 0
        x_max = 80
    if col >= 40:    
        x_min = col - 40
        if col + 40 >= x_shape:
            x_max = x_shape
            x_min = x_max - 80
        else:
            x_max = col + 40
    
    bounds = [y_min, y_max, x_min, x_max]
    test_areas.append(bounds)
    
test_areas_set = set(tuple(x) for x in test_areas)
test_areas = [list(x) for x in test_areas_set]

#%%

# Rather than using a moving window and predicting on millions of images,
# this method only predicts based on the coordinates of K means minority class

def test_windows(bound):
    window = np.arange(im_bands * width * height).reshape(im_bands, width, height)
    for i in bound:
        window[0] = im_tensor[0, bound[0]:bound[1], bound[2]:bound[3]]
        window[1] = im_tensor[0, bound[0]:bound[1], bound[2]:bound[3]]
        window[2] = im_tensor[0, bound[0]:bound[1], bound[2]:bound[3]]
    window = window.reshape([-1, im_bands, width, height])
    window = window.transpose([0,2,3,1])
    window = window / 255
    
    return window


# Define paint box function, draws bounding box on image of positive predictions
def paint_box(x, y, acc, border_width):   
    for i in range(80):
        for ch in range(3):
            for th in range(border_width):
                im_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(border_width):
                im_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(border_width):
                im_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(border_width):
                im_tensor[ch][y+th+80][x+i] = -1

#%%

im_tensor = im_tensor.transpose(2,0,1)


#%%

# Define the threshold for consideration relative to the mask layer used in narrowing image area
num_pix = width * height * im_bands
mask_threshold = 0.9 * num_pix


#%%

start = datetime.now()

coordinates = []

for idx, i in enumerate(test_areas):
    sys.stdout.write('\r{}%  '.format(round(idx/len(test_areas), 3)))
    window = test_windows(i)
    result = model.predict(window)
    if result[0][1] > 0.9:
        if np.count_nonzero(window) > mask_threshold:
            
            coords = [[i[2], i[0]], [i[3], i[0]], [i[3], i[1]], [i[2], i[1]]]
            polygon = Polygon(coords)
            
            
            if len(coordinates) == 0:
                coordinates.append([coords, result])
            
            else:
                iou_list = []
                for (j, item) in enumerate(coordinates):
                    if abs(coords[0][0] - item[0][0][0]) >= 80:
                        pass
                    else:
                        poly_j = Polygon(item[0])
                        intersection = polygon.intersection(poly_j)
                        iou = intersection.area / 6400
                        iou_list.append(iou)
                
                if len(iou_list) == 0 and [coords, result] not in coordinates:
                    coordinates.append([coords, result])
                    
                elif max(iou_list) < 0.1 and [coords, result] not in coordinates:
                    coordinates.append([coords, result])
                    
                else:
                    deleted_items = []
                    for (k, item) in enumerate(coordinates):
                        poly_k = Polygon(item[0])
                        intersection = polygon.intersection(poly_k)
                        iou = intersection.area / 6400
                        
                        if iou > 0.1 and result[0][1] > item[1][0][1]:
                            deleted_items.append(item)
                            if [coords, result] not in coordinates:
                                coordinates.append([coords, result])
                        else:
                            pass
                    coordinates[:] = [e for e in coordinates if e not in deleted_items]
                            
                                
end = datetime.now()
print('total time = {}'.format(end-start))                               
                                
#%%

# Further eliminate predictions that are overlapping and have a lower prediction score
# Keeping only one prediction per object, with highest score....ideally
deleted_items = []   
for (i, item) in enumerate(coordinates):
    poly_i = Polygon(item[0])
    for (j, comp) in enumerate(coordinates):
        if abs(item[0][0][0] - comp[0][0][0]) < 80:
            poly_j = Polygon(comp[0])
            intersection = poly_i.intersection(poly_j)
            iou = intersection.area / 6400
            
            if iou > 0.1 and item != comp:
                
                if item[1][0][1] > comp[1][0][1]:
                    deleted_items.append(comp)
                else:
                    deleted_items.append(item)
                coordinates = [e for e in coordinates if e not in deleted_items]


# Paint the bounding boxes on image
for e in coordinates:
    paint_box(e[0][0][0], e[0][0][1], e[1][0][1], 5)
    

#%%

# Reshape and view results
im_tensor = im_tensor.transpose(1, 2, 0)

plt.figure(1, figsize = (15, 30))

plt.subplot(3, 1, 1)
plt.imshow(im_tensor)

plt.show()


#%%

# Convert positive predictions to WKT strings to be used in importing to GIS software
wkt_strings = []
for i in coordinates:
    x_min = (i[0][0][0] * im_x_res) + left
    y_min = top - (i[0][0][1] * im_y_res)
    x_max = (i[0][1][0] * im_x_res) +left
    y_max = top - (i[0][2][1] * im_y_res)
    
    wkt = 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(
        x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, x_min, y_min)
    wkt_strings.append(wkt)


# Export WKT to desired location
df = pd.DataFrame(wkt_strings, columns=['wkt'])
df.to_csv('.../test_wkt.csv')


#%%
