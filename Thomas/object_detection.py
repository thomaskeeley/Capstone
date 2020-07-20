#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:01:55 2020

@author: thomaskeeley
"""


#%%

import numpy as np
import pandas as pd
import sys

from datetime import date

from shapely.geometry import Polygon

from tensorflow import keras

import fiona
import rasterio
import rasterio.features
import rasterio.mask

from matplotlib import pyplot as plt

from train_model import TrainModel
        
        
#%%

class ObjectDetection:
    
    def __init__(self, image_dir, mask_dir, out_dir, step, prediction_threshold,
                 pos_chip_dir, neg_chip_dir, chip_width, chip_height, 
                 augment_pos_chips, augment_neg_chips, save_model, import_model):
        """
        
        Parameters
        ----------
        image_dir : Directory
            DESCRIPTION: Filepath to image the be used in object detection
        mask_dir : Directory
            DESCRIPTION: Filepath to optional mask layer to be used to decreasing consideration area for object detection
        out_dir : Directory
            DESCRIPTION: Filepath to desired output location of results
        step : Integer
            DESCRIPTION: Value that defines the number of pixels moving window predictions jumps by
        prediction_threshold : Float
            DESCRIPTION: Value between 0-1 that defines the consideration minimum for positive prediction
        pos_chip_dir : Directory
            DESCRIPTION: Filepath to positive image chips(contain target object for detection)
        neg_chip_dir : Directory
            DESCRIPTION: Filepath to negative image chips(do not contain target object)
        chip_width : Integer
            DESCRIPTION: Desired output width of output training chip
        chip_height : Integer
            DESCRIPTION: Desired output height of output training chip
        augment_pos_chips : True or False
            DESCRIPTION: Option to create additional training data for positive chips through augmentation
        augment_neg_chips : True or False
            DESCRIPTION: Option to create additional training data for negative chips through augmentation.
        save_model : True or False
            DESCRIPTION: Option to save the model to current directory
        import_model : Directory
            DESCRIPTION: Filepath to saved model
        
        Notes
        -------
        The purpose of this module is to incorporate training data and trained model to conduct object detection in an image.
        A built in function conducts predictions across the image using a "moving window" where "windows" that have 
        a prediction > user-defined predction_threshold are converted into coordinates and stored in a list.
        Next, the positive prediction coordinates are filtered to produce only one prediction per potential object.
        This is done by analyzing overlapping bounding boxes and keeping the highest scoring prediction.
        The final results are then referenced back to the geographic attribution of the image and converted into 
        well known text(WKT) strings that can be imported as geographic data into GIS software.
        
        """
        self.image_filepath = image_dir 
        self.mask_filepath = mask_dir
        self.out_dir = out_dir
        self.step = step
        self.prediction_threshold = prediction_threshold
        self.width = chip_width
        self.height = chip_height
        self.chip_pix = chip_width * chip_height * 3
        self.mask_threshold = self.chip_pix * 0.9
        self.import_model = import_model
        self.train_model = TrainModel(pos_chip_dir, neg_chip_dir, chip_width, chip_height, 
                                      augment_pos_chips, augment_neg_chips, save_model)
        
        
          
    def get_image_data(self):
        """
        
        Returns
        -------
        im_bands : Integer
            DESCRIPTION: Number of layers or bands in image
        im_width : Integer
            DESCRIPTION: Image width (pixels)
        im_height : Integer
            DESCRIPTION: Image height (pixels)
        left : Float
            DESCRIPTION: Western geographic bounding line
        bottom : Float
            DESCRIPTION: Southern geographic bounding line
        right : Float
            DESCRIPTION: Eastern geographic bounding line
        top : Float
            DESCRIPTION: Northern geographic bounding line
        im_x_res : Float
            DESCRIPTION: Image x resolution (pixel size in ground measurement)
        im_y_res : Float
            DESCRIPTION: Image y resolution (pixel size in ground measurement)
        crs : String
            DESCRIPTION: Image coordinate reference system

        """
        im = rasterio.open(self.image_filepath)
        
        im_bands = im.count
        
        im_width = im.width
        im_height = im.height
        
        left, bottom, right, top = im.bounds
        
        im_x_res = (right-left)/im_width
        im_y_res = (top-bottom)/im_height  
        
        crs = str(im.crs).split(':')[1] 
        
        return im_bands, im_width, im_height, left, bottom, right, top, im_x_res, im_y_res, crs
    
    
    def create_tensor(self):
        """
        
        Returns
        -------
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        """
        print('\n >>> CREATING TENSOR')
        im = rasterio.open(self.image_filepath)
        
        im_tensor = im.read()
        im_tensor = im_tensor.transpose(1, 2, 0)   
        
        if self.mask_filepath:
            with fiona.open(self.mask_filepath, "r") as shapefile:
                geoms = [feature["geometry"] for feature in shapefile]
            
                out_image, out_transform = rasterio.mask.mask(im, geoms, invert=True)
                
                im_tensor = out_image.transpose(1, 2, 0)
        print('\n     Complete')
        return im_tensor
    
       
    def moving_window(self, x, y, im_tensor):
        """
        
        Parameters
        ----------
        x : Integer
            DESCRIPTION: Current x position in image
        y : Integer
            DESCRIPTION: Current y position in image
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        Returns
        -------
        window : Array of uint8
            DESCRIPTION: Image patch from larger image to run prediction on

        """
        window = np.arange(3 * self.width * self.height).reshape(3, self.width, self.height)
        for i in range(self.width):
            for j in range(self.height):
                window[0][i][j] = im_tensor[0][y+i][x+j]
                window[1][i][j] = im_tensor[1][y+i][x+j]
                window[2][i][j] = im_tensor[2][y+i][x+j]
        window = window.reshape([-1, 3, self.width, self.height])
        window = window.transpose([0,2,3,1])
        window = window / 255
        return window     
    
        
    def detect(self, im_tensor):
        """
        
        Parameters
        ----------
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        Returns
        -------
        coordinates : List
            DESCRIPTION: Employing the moving window function, the windows that have > prediction threshold values
            are converted into coordinates that reference position in image array and stored in a list.

        """
        step = self.step; coordinates = []
        im_width, im_height = self.get_image_data()[1:3]
        if self.import_model:
            model = keras.models.load_model(self.import_model)
        else:
            model = self.train_model.execute()
        
        print('\n >>> CONDUCTING OBJECT DETECTION')
        for y in range(int((im_height-(20-step))/step)):
            for x in range(int((im_width-(20-step))/step) ):
                window = self.moving_window(x*step, y*step, im_tensor)
                if np.count_nonzero(window) == 0:
                    x += 20
                else:
                    result = model.predict(window)
                    sys.stdout.write('\r     {}%'.format(round(y*step/im_height*100, 1)))
                    if result[0][1] > self.prediction_threshold:
                        if np.count_nonzero(window) > self.mask_threshold:
                            x_min = x*step
                            y_min = y*step
                            
                            x_max = x_min + 20
                            y_max = y_min + 20
                            
                            coords = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                            
                            coordinates.append([coords, result])
        print('\n     Complete')            
        return coordinates
        
        
    def trim_results(self, coordinates):
        """
        
        Parameters
        ----------
        coordinates : List
            DESCRIPTION: Collection of image array coordinates that a positive predictions for object detection

        Returns
        -------
        trimmed_coordinates : List
            DESCRIPTION: Coordinates are analyzed relative to overlapping coordinate bounds. The highest prediction value
            is kept and the others are discarded. 

        """
        print('\n >>> TRIMMING RESULTS')
        deleted_items = []   
        for (i, item) in enumerate(coordinates):
            poly_i = Polygon(item[0])
            for (j, comp) in enumerate(coordinates):
                if abs(item[0][0][0] - comp[0][0][0]) < 20:
                    poly_j = Polygon(comp[0])
                    intersection = poly_i.intersection(poly_j)
                    iou = intersection.area / 400
                    
                    if iou > 0.1 and item != comp:
                        
                        if item[1][0][1] > comp[1][0][1]:
                            deleted_items.append(comp)
                        else:
                            deleted_items.append(item)
                        trimmed_coordinates = [e for e in coordinates if e not in deleted_items]
        print('\n     Complete')
        return trimmed_coordinates
        
        
        
    def create_geo_output(self, trimmed_coordinates):
        """
        
        Parameters
        ----------
        trimmed_coordinates : List
            DESCRIPTION: Final collection of positive predictions

        Returns
        -------
        wkt_strings : List
            DESCRIPTION: Image array coordinates are transformed back to original geographic coordinates of input image.
            Values are stored as strings that are capable of being loaded as geographic features in GIS software.

        """
        print('\n >>> CREATING GEOGRAPHIC OUTPUT')
        im_bands, im_width, im_height, left, bottom, right, top, im_x_res, im_y_res, crs = self.get_image_data()
        wkt_strings = []
        for i in trimmed_coordinates:
            x_min = (i[0][0][0] * im_x_res) + left
            y_min = top - (i[0][0][1] * im_y_res)
            x_max = (i[0][1][0] * im_x_res) +left
            y_max = top - (i[0][2][1] * im_y_res)
            
            wkt = 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, x_min, y_min)
            wkt_strings.append(wkt)
        print('\n     Complete')    
        return wkt_strings
        
        
    
    def paint_box(self, x, y, border_width, im_tensor):   
        """
        
        Parameters
        ----------
        x : Integer
            DESCRIPTION: x position of final coordinates predictions
        y : Integer
            DESCRIPTION y position of final coordinates predictions
        border_width : Integer
            DESCRIPTION: Input for visual border thickness to be "painted" on image
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        Returns
        -------
        None, this method "paints" bounding boxes on original image to display prediction results

        """
        for i in range(self.width):
            for ch in range(3):
                for th in range(border_width):
                    im_tensor[ch][y+i][x-th] = -1
    
        for i in range(self.width):
            for ch in range(3):
                for th in range(border_width):
                    im_tensor[ch][y+i][x+th+20] = -1
            
        for i in range(self.width):
            for ch in range(3):
                for th in range(border_width):
                    im_tensor[ch][y-th][x+i] = -1
            
        for i in range(self.width):
            for ch in range(3):
                for th in range(border_width):
                    im_tensor[ch][y+th+20][x+i] = -1
    
    
    def show_results(self, trimmed_coordinates, im_tensor):
        """
        
        Parameters
        ----------
        trimmed_coordinates : List
            DESCRIPTION: Final collection of positive predictions
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        Returns
        -------
        None, this method executes the paint_box method and plots the results

        """
        print('\n >>> SHOWING RESULTS')
        
        for e in trimmed_coordinates:
            self.paint_box(e[0][0][0], e[0][0][1], 2, im_tensor)
        
        im_tensor = im_tensor.transpose(1, 2, 0)
        plt.figure(1, figsize = (15, 30))
        
        plt.subplot(3, 1, 1)
        plt.imshow(im_tensor)
        print('\n     Check Your Plots')
        plt.show()
        
        
    def execute(self):
        print('\n ~~~EXECUTION IN-PROGRESS~~~')
        im_tensor = self.create_tensor()
        im_tensor = im_tensor.transpose(2,0,1)
        coordinates = self.detect(im_tensor)
        trimmed_coordinates = self.trim_results(coordinates)
        wkt_strings = self.create_geo_output(trimmed_coordinates)
        crs = self.get_image_data()[-1]
        self.show_results(trimmed_coordinates, im_tensor)
        
        print('\n >>> EXPORTING DATA')
        df = pd.DataFrame(wkt_strings, columns=['wkt'])
        df.to_csv(self.out_dir + 'obj_det_wkt-{}_{}.csv'.format(crs,date.today().strftime('%Y%m%d')))
        print('\n ~~~EXECUTION COMPLETE~~~')
        
#%%      
        

        
