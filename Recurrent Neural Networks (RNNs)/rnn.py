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
training_set_scaled = sc.fit_transform(training_set)

#DATASTRUCTURE TO REPRESENT WAHT THE RNN NEEDS TO REMEMBER FOR PREDICTION
# 'NUMBER OF TIME STEPS'

#->CREATING A DATA STRUCTURE WITH 60 TIMESTEPS AND 1 OUTPUT
#->LOOKS BACK AT THE PREVIOUS 60 DAYS FOR PREDICTING CURRENT STOCK PRICE
#X_TRAIN: 60 PREVIOUS DAYS FOR EVERY TIME
#Y_TRAIN: NEXT STOCK PRICE FOR EVERY TIME
X_train = []
Y_train = []

for i in range(60, 1258):
    #APPEND INTO X_TRAIN: ROWS, COLUMN
    X_train.append(training_set_scaled[i-60:i, 0]) #MEMORY
    Y_train.append(training_set_scaled[i, 0]) #PREDICTIONS

#CONVERT TO NUMPY ARRAY
X_train, Y_train = np.array(X_train), np.array(Y_train)


#ADD THE NEW DIMENSION IN THE NUMPY ARRAY: RESHAPING
#WE CAN ADD MORE INDICATORS TO PREDICT THE GOOGLE STOCK PRICE
#-> CURRENT NUMPY ARRAY AND THE NEW DIMENSION (INDICATOR)
#-> BATCH_SIZE (TOTAL OBSERVATIONS), TIMESTEPS, INPUT_DIM (NO OF INDICATORS)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# PART - 2 BUILDING THE RNN

#IMPORTING KERAS LIBRARIES AND PACKAGES
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout



#INITIALISE THE RNN
regressor = Sequential() #CONTINOUS PREDICTION: REGRESSION


#FIRST LSTM LAYER
regressor.add(LSTM(
    units = 50,
    return_sequences=True, #SEVERAL LSTM LAYERS
    input_shape = (X_train.shape[1], 1)  #TIMESTEMPS & INDICATORS
    ))

#DROPOUT REGULARISATION: AVOID OVERFITTING
regressor.add(Dropout(
    rate = 0.2 #RATE OF NEURONS TO DROP OR IGNORE DURING TRAINING
  ))

#SECOND LSTM LAYER
regressor.add(LSTM(
    units = 50, #INPUTS SPECCIFIED BY NUMBER OF NEURONS IN PREVIOUS LAYER
    return_sequences=True, #SEVERAL LSTM LAYERS
    ))
#DROPOUT REGULARISATION
regressor.add(Dropout(
    rate = 0.2 #RATE OF NEURONS TO DROP OR IGNORE DURING TRAINING
  ))


#THIRD LSTM LAYER
regressor.add(LSTM(
    units = 50, #INPUTS SPECCIFIED BY NUMBER OF NEURONS IN PREVIOUS LAYER
    return_sequences=True, #SEVERAL LSTM LAYERS
    ))


#DROPOUT REGULARISATION
regressor.add(Dropout(
    rate = 0.2 #RATE OF NEURONS TO DROP OR IGNORE DURING TRAINING
  ))


#FOURTH LSTM LAYER: OUTPUT
regressor.add(LSTM(
    units = 50, #INPUTS SPECCIFIED BY NUMBER OF NEURONS IN PREVIOUS LAYER
    return_sequences=False, #FINAL LSTM LAYER
    ))

#DROPOUT REGULARISATION
regressor.add(Dropout(
    rate = 0.2 #RATE OF NEURONS TO DROP OR IGNORE DURING TRAINING
  ))


#FINAL - OUTPUT LAYER
regressor.add(Dense( #FULLY CONNECTED TO PREVIOUS LAYER
    units = 1 #ONE OUTPUT VALUE
    ))

#COMPILING THE RNN
regressor.compile(
    optimizer = "adam", #ALWAYS A SAFE CHOICE; RNN, ANN
    loss  = "mean_squared_error" #MSE FOR REGRESSION
    )

#FITTING THE RNN TO TRAINING SET
regressor.fit(
    X_train,
    Y_train,
    epochs = 100,
    batch_size =  32
    )

# PART - 3: MAKING THE PREDICTIONS AND VISUALISING THE RESULTS
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price  = dataset_test.iloc[:, 1:2].values #ALL ROWS; 1ST COLUMN

# GET THE PREDICTED STOCK PRICE
#->NEED ENTIRE DATASET TO GET PREVIOUS 60 DAYS FOR EACH DAY
#->CONCATENATE THE ORIGINAL DATAFRAMES
dataset_total = pd.concat(
    (dataset_train['Open'],  dataset_test['Open']),  
    axis = 0 #ROWS; NOT COLUMNS
)

#GET INPUTS [60 PREVIOUS DAY PRICES FOR JAN]
#1ST FINANCIAL DAY OF JAN <-> 60 TO LAST FINANCIAL DAY OF JAN - 60
inputs = dataset_total[
    len(dataset_total)-len(dataset_test)-60:].values

 #RESHAPE THE INPUTS
inputs = np.reshape(-1,  +1)
 
 #SCALE INPUTS
inputs = sc.transform(inputs) #NO FITTING ON TESTING DATA
 
 #3D FORMAT : BATCH_SIZE, TIMESTEPS, INDICATORS
X_test = []

#20 FINANCIAL DAYS OF JJANUAR
for i in range(60, 80):
     #APPEND INTO X_TRAIN: ROWS, COLUMN
     X_test.append(inputs[i-60:i, 0]) #MEMORY

X_test = np.array(X_test) 

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#MAKE THE PREDICTIONS
#->INPUT = 60 PREVIOUS DAYS
#->OUTPUT = NEXT DAY'S PREDICTED STOCK PRICE
predicted_stock_price = regressor.predict(X_test)

#INVERSE THE SCALING OF PREDICTIONS
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#VISUALISE THE RESULTS
plt.plot(real_stock_price,
         color = 'red',
         label = "Real Google Stock Price")
plt.plot(predicted_stock_price,
         color = 'blue',
         label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time: January")
plt.ylabel("Google Stock Price")
plt.legend()
