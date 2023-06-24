#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 01:17:28 2023

@author: kpandey
"""

# PART 1 - DATA PREPROCESSING

#IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#IMPORTING THE TRAINING SET

    #READ AS DATAFRAME
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
    #NUMPY ARRAY WITH ALL ROWS AND ONLY THE 1ST COLUMN
training_set = dataset_train.iloc[:,1:2].values


#USE NORMALISATION FOR FEATURE SCALING [minmax scaling]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_Scaled = sc.fit_transform(training_set)



# PART - 2 BUILDING THE RNN

# PART - 3: MAKING THE PREDICTIONS AND VISUALISING THE RESULTS
