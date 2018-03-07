# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:07:21 2018

@author: bsimmons
"""

import cv2
import numpy as np
import glob
import sys
import keras
import h5py
from keras.models import Sequential
#from keras.layers import LSTM, Dense
from keras.layers import Dense
from sklearn.model_selection import train_test_split

print ('Loading training data...')
e0 = cv2.getTickCount()

# load training data
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')
training_data = glob.glob('training_data/*.npz')


# if no data, exit
if not training_data:
    print ("No training data in directory, exit")
    sys.exit()
for single_npz in training_data:
    with np.load(single_npz) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :]
y = label_array[1:, :]
print ('Image array shape: ', X.shape)
print ('Label array shape: ', y.shape)

e00 = cv2.getTickCount()
time0 = (e00 - e0)/ cv2.getTickFrequency()
print ('Loading image duration:', time0)


# train test split, 7:3
# X_train, X_test, y_train, y_test
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

# set start time
e1 = cv2.getTickCount()

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 38400))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output  layer, 
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(loss=keras.losses.binary_crossentropy,
                   optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
# Compiling the ANN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(train, train_labels, batch_size = 10, epochs = 100)


# set end time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print ('Training duration:', time)


# Predicting the Test set results
y_pred = classifier.predict(test)
y_pred = (y_pred > 0.5)

test_rate = np.mean(y_pred == test_labels)
print ('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

# save model
classifier.save_weights('Saved_Weights_ANN.h5')