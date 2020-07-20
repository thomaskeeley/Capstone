#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:14:35 2020

@author: thomaskeeley
"""


#%%


import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from create_training_data import CreateTrainingChips

#%%


class TrainModel:
    
    def __init__(self, pos_chip_dir, neg_chip_dir, chip_width, chip_height, augment_pos_chips, augment_neg_chips, save_model):
        """
        
        Parameters
        ----------
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
            DESCRIPTION: Option to create additional training data for negative chips through augmentation
        save_model : True or False
            DESCRIPTION: Option to save model to output directory
            
        Notes
        -------
        The purpose of this module is to first split training tensors into a train/test split.
        Next, a keras sequential model is defined, compiled, and fit to the training data.
        This module may be customized to integrate pre-trained models or add additional layers.
        The output is a trained model that can be saved to a local directory and imported later.
        """
        self.training_data = CreateTrainingChips(pos_chip_dir, neg_chip_dir, chip_width, chip_height, 
                                                 augment_pos_chips, augment_neg_chips)
        self.chips, self.labels = self.training_data.execute()
        self.save_model = save_model

            
    def train_test(self):
        """
        
        Returns
        -------
        X_train
        X_test
        y_train
        y_test
            DESCRIPTION: Split of training data, shuffled by random indexes, and normalized

        """
        X = self.chips

        y = self.labels

        # shuffle all indexes
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        
        X = X[indexes].transpose([0,2,3,1])
        y = y[indexes]
        
        # normalization
        X = X / 255
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        
        return X_train, X_test, y_train, y_test

        
    def execute(self):
        """
        
        Returns
        -------
        model : 
            DESCRIPTION: Trained model based on user defined parameters, saved if defined in model_save parameter

        """
        print('\n >>> TRAINING MODEL')
        X_train, X_test, y_train, y_test = self.train_test()
        
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
        
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, shuffle=True, verbose=1)
        print('\n     Training Complete')
        print('\n     Model Accuracy = {}'.format(model.evaluate(X_test, y_test, verbose=2)[1]))
        
        if self.save_model == True:
            model.save('model.h5')
     
        return model
        
        

#%%
      
