# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:33:34 2019

@author: porush
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D


import os

path = "mirror"
X = []
y = []
#train for chair
def create_test_data(path):
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append([1,0,0,0])
create_test_data(path)

path = "Table"

def create_test_data(path):
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append([0,1,0,0])
create_test_data(path)

path = "chair"

def create_test_data(path):
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append([0,0,1,0])
create_test_data(path)

path = "container"

def create_test_data(path):
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append([0,0,0,1])
create_test_data(path)

X = np.array(X).reshape(-1, 80,80,1)
y = np.array(y)
    
#Normalize data
X = X/255.0

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 4 output units:
model.add(Dense(4, activation='softmax'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.00001)
 
path = "test"
X_test = []
id_line = []
def create_test1_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)
create_test1_data(path)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255

predictions = model.predict(X_test)

import operator
def find(x):
    index, value = max(enumerate(x), key=operator.itemgetter(1))
    return index+1
prediction_new = []
for i in range(len(predictions)):
    number = find(predictions[i])
    prediction_new.append(number)
    
confusion_matrix = np.zeros((4,4))

submission_df = pd.DataFrame({'id':id_line, 'label':prediction_new})

submission_df.to_csv("submission.csv", index=False)
#Confusion matrix 

for i in range(4):
    if i==0:
        for j in range(12):
            if prediction_new[j]==1:
                confusion_matrix[2][0]+=1
            if prediction_new[j]==2:
                confusion_matrix[2][1]+=1
            if prediction_new[j]==3:
                confusion_matrix[2][2]+=1
            if prediction_new[j]==4:
                confusion_matrix[2][3]+=1
    if i==1:
        for j in range(12,24):
            if prediction_new[j]==1:
                confusion_matrix[0][0]+=1
            if prediction_new[j]==2:
                confusion_matrix[0][1]+=1
            if prediction_new[j]==3:
                confusion_matrix[0][2]+=1
            if prediction_new[j]==4:
                confusion_matrix[0][3]+=1
    if i==2:
        for j in range(24,37):
            if prediction_new[j]==1:
                confusion_matrix[1][0]+=1
            if prediction_new[j]==2:
                confusion_matrix[1][1]+=1
            if prediction_new[j]==3:
                confusion_matrix[1][2]+=1
            if prediction_new[j]==4:
                confusion_matrix[1][3]+=1
    if i==3:
        for j in range(37,49):
            if prediction_new[j]==1:
                confusion_matrix[3][0]+=1
            if prediction_new[j]==2:
                confusion_matrix[3][1]+=1
            if prediction_new[j]==3:
                confusion_matrix[3][2]+=1
            if prediction_new[j]==4:
                confusion_matrix[3][3]+=1
