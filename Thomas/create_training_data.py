#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:03:29 2020

@author: thomaskeeley
"""


from PIL import Image
import glob
import os
import numpy as np
from keras.utils import np_utils

#%%


class CreateTrainingChips:
    
    def __init__(self, pos_chip_dir, neg_chip_dir, chip_width, chip_height, augment_pos_chips, augment_neg_chips):
        """Create training tensors from image chips.
        
        Note:
            The purpose of this module is to ingest image chips manually created by user in GIS that are broken down
            into two categories: 
                1. chips that contain target object 
                2. chips that are random background/negative labels.
        
        After the image is cropped to the desired dimension, it is transformed into tensors to be used as 
        training data in a Deep Learning model.
        The output is a collection of training tensors and associated label (0,1)
                
        Parameters
        ----------
        pos_chip_dir : Directory
            Filepath to positive image chips(contain target object for detection)
        neg_chip_dir : Directory
            Filepath to negative image chips(do not contain target object)
        chip_width : Integer
            Desired output width of output training chip
        chip_height : Integer
            Desired output height of output training chip
        augment_pos_chips : True or False
            Option to create additional training data for positive chips through augmentation
        augment_neg_chips : True or False
            Option to create additional training data for negative chips through augmentation

        """
        
        self.pos_dir = pos_chip_dir
        self.neg_dir = neg_chip_dir
        self.width = chip_width
        self.height = chip_height
        self.pos_augment = augment_pos_chips
        self.neg_augment = augment_neg_chips
        
    def create_pos_chips(self):
        """
        
        Returns
        -------
        pos_chips : List
            DESCRIPTION: Contains positive image chips

        """
        os.chdir(self.pos_dir)
        pos_filenames = [i for i in glob.iglob('*.tif')]
        pos_chips = []
        
        for file in pos_filenames:
            
            im = Image.open(file)
            
            chip_width, chip_height = im.size
            
            width_min = round((chip_width - self.width) / 2)
            width_max = width_min + self.width
            
            height_min = round((chip_height - self.height) / 2)
            height_max = height_min + self.height
            
            im = im.crop((width_min, height_min, width_max, height_max))
            
            pos_chips.append(im)
            
            if self.pos_augment == True:
                
                pos_chips.append(im.transpose(Image.FLIP_LEFT_RIGHT))
                pos_chips.append(im.transpose(Image.FLIP_TOP_BOTTOM))
                pos_chips.append(im.rotate(90))
                pos_chips.append(im.rotate(180))
            
        return pos_chips
        
        

    def create_neg_chips(self):
        """
        
        Returns
        -------
        neg_chips : List
            DESCRIPTION: Contains negative image chips

        """
        os.chdir(self.neg_dir)
        neg_filenames = [i for i in glob.iglob('*.tif')]
        neg_chips = []
        
        for file in neg_filenames:
            
            im = Image.open(file)
            
            chip_width, chip_height = im.size
            
            width_min = round((chip_width - self.width) / 2)
            width_max = width_min + self.width
            
            height_min = round((chip_height - self.height) / 2)
            height_max = height_min + self.height
            
            im = im.crop((width_min, height_min, width_max, height_max))
            
            neg_chips.append(im)
            
            if self.neg_augment == True:
                
                neg_chips.append(im.transpose(Image.FLIP_LEFT_RIGHT))
                neg_chips.append(im.transpose(Image.FLIP_TOP_BOTTOM))
                neg_chips.append(im.rotate(90))
                neg_chips.append(im.rotate(180))
            
        return neg_chips
        
    
    
    def create_training_data(self, pos_chips, neg_chips):
        """
        
        Returns
        -------
        chips : Array of uint8
            Collection of training tensors
        labels : Array of uint8
            Categorical labels for training tensors (0,1)
        
        
        >>> chips[0]
        array([[[ 53,  54,  51, ...,  54,  54,  53],
                [ 52,  53,  54, ...,  52,  55,  52],
                [ 52,  53,  52, ...,  53,  51,  52],
                ...,
                [ 51,  49,  50, ...,  49,  49,  50],
                [ 48,  48,  50, ...,  49,  47,  47],
                [ 50,  51,  48, ...,  49,  49,  49]]], dtype=uint8)
        """
        chips = []
        labels = []
        for chip in pos_chips:
            imarray = np.array(chip).astype('uint8')
            imarray = np.moveaxis(imarray, -1, 0)
            chips.append(imarray)
            labels.append(1)
            
        for chip in neg_chips:
            imarray = np.array(chip).astype('uint8')
            imarray = np.moveaxis(imarray, -1, 0)
            chips.append(imarray)
            labels.append(0)
            
        chips = np.array(chips).astype('uint8')
        labels = np_utils.to_categorical(labels, 2)  
        
        return chips, labels
        
            
        
    def execute(self):
        # print('CREATING TRAINING DATA...')
        pos_chips = self.create_pos_chips()
        neg_chips = self.create_neg_chips()
        
        chips, labels = self.create_training_data(pos_chips, neg_chips)
        
        return chips, labels




