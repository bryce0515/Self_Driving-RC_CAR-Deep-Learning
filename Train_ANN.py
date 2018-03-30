"""
Created on Wed Jan 17 13:07:21 2018

@author: bsimmons
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import keras
#import h5py

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

print ('Loading training data...')
e0 = cv2.getTickCount()

num_epochs = 30
batch_size = 10
image_size = 54000
# load training data
image_array = np.zeros((1, image_size))
label_array = np.zeros((1, 4), 'float')
training_data = glob.glob('training_data_ANN/*.npz')


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
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.3)


## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

# set start time
e1 = cv2.getTickCount()

# Initialising the ANN
classifier = Sequential()

#X = X.reshape((1,38400,len(image_array))
# Adding the input layer 
#classifier.add(LSTM(32, input_shape = (38400,image_array[0])))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = image_size))
classifier.add(Dropout(0.5))

# Adding the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.5))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.5))

# Adding the third hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.5))

# Adding the output  layer, 
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
#classifier.compile(loss=keras.losses.binary_crossentropy,
#                   optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),metrics=['accuracy'])
# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Viewing model_configuration

classifier.summary()
classifier.get_config()
classifier.layers[0].get_config()
classifier.layers[0].input_shape
classifier.layers[0].output_shape
classifier.layers[0].get_weights()
np.shape(classifier.layers[0].get_weights()[0])
classifier.layers[0].trainable

# Fitting the ANN to the Training set
#hist = classifier.fit(train, train_labels, batch_size = 16, epochs = num_epochs, validation_data=(test, test_labels))

# Training with callbacks
from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

hist = classifier.fit(train, train_labels, batch_size = batch_size,verbose=1, epochs = num_epochs, validation_data=(test, test_labels),callbacks=callbacks_list)


# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

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
classifier.save('Saved_Weights.xml')
