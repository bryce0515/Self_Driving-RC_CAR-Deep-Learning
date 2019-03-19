from __future__ import print_function
#import glob
#import sys
import time
import numpy as np
import matplotlib.pylab as plt
from scipy import misc

#import keras
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad,Adadelta, Adamax,Nadam
from keras.models import Sequential,load_model
from keras.layers import Lambda, concatenate
from keras import Model
from keras import callbacks
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gc

import pandas as pd

# results = pd.read_csv (r'C:\Users\bsimmons\Documents\self_driving_car\Documents\VGG16_1.csv')
# results = np.array(results)
#acc = results[:,1]
#loss = results[:,2]
            
# def multi_gpu_model(model, gpus):
#   if isinstance(gpus, (list, tuple)):
#     num_gpus = len(gpus)
#     target_gpu_ids = gpus
#   else:
#     num_gpus = gpus
#     target_gpu_ids = range(num_gpus)

#   def get_slice(data, i, parts):
#     shape = tf.shape(data)
#     batch_size = shape[:1]
#     input_shape = shape[1:]
#     step = batch_size // parts
#     if i == num_gpus - 1:
#       size = batch_size - step * i
#     else:
#       size = step
#     size = tf.concat([size, input_shape], axis=0)
#     stride = tf.concat([step, input_shape * 0], axis=0)
#     start = stride * i
#     return tf.slice(data, start, size)

#   all_outputs = []
#   for i in range(len(model.outputs)):
#     all_outputs.append([])

#   # Place a copy of the model on each GPU,
#   # each getting a slice of the inputs.
#   for i, gpu_id in enumerate(target_gpu_ids):
#     with tf.device('/gpu:%d' % gpu_id):
#       with tf.name_scope('replica_%d' % gpu_id):
#         inputs = []
#         # Retrieve a slice of the input.
#         for x in model.inputs:
#           input_shape = tuple(x.get_shape().as_list())[1:]
#           slice_i = Lambda(get_slice,
#                            output_shape=input_shape,
#                            arguments={'i': i,
#                                       'parts': num_gpus})(x)
#           inputs.append(slice_i)

#         # Apply model on slice
#         # (creating a model replica on the target device).
#         outputs = model(inputs)
#         if not isinstance(outputs, list):
#           outputs = [outputs]

#         # Save the outputs for merging back together later.
#         for o in range(len(outputs)):
#           all_outputs[o].append(outputs[o])

#   # Merge outputs on CPU.
#   with tf.device('/cpu:0'):
#     merged = []
#     for name, outputs in zip(model.output_names, all_outputs):
#       merged.append(concatenate(outputs,
#                                 axis=0, name=name))
#     return Model(model.inputs, merged)

def subtract_median(arr):
    return arr - np.median(arr)
    
def norm_image(img):
    _min = 0
    _max = 1
    img_lin = 10**(img/10) 
    X = np.log10(img_lin)
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (_max - _min) + _min
    return X_scaled


""" combine multiple files """
def combine(x1,x2,y1,y2):
    xjoin = np.concatenate((x1,x2),axis=0)
    yjoin = np.concatenate((y1,y2),axis=0)
    return xjoin,yjoin
    
def norm_subtract(image):
    imag = []
    for x in range(len(image)):
        x_t = subtract_median(image[x])
        x_t = norm_image(x_t)
        ##remove shoulders
        imag.append(x_t)
    return np.array(imag)
    
"""
#***********Resizing+ + Combining Images/Labels + Pre-Processing **************
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

right = np.load('C:/Users/bsimmons/Documents/self_driving_car/Training_Data/right_images_updated.npy')
right_label = np.load('right_label.npy')

foward = np.load('forward_images_updated.npy')
foward_label = np.load('foward_label.npy')

x_t, y_t = combine(right, foward, right_label, foward_label)

x_tr = []
for n in range(len(x_t)):
    x_tr.append(misc.imresize(x_t[n],(30,90),interp='bilinear'))
    
x_tr = np.array(x_tr)

left = np.load('left_images_updated.npy')
left_label = np.load('left_label.npy')

reverse = np.load('reverse_images.npy')
reverse_label = np.load('reverse_label.npy')

x_t1, y_t1 = combine(left, reverse, left_label, reverse_label)

x_tr2 = []
for n in range(len(x_t1)):
    x_tr2.append(misc.imresize(x_t1[n],(30,90),interp='bilinear'))
    
x_tr2 = np.array(x_tr2)  
x_t2, y_t2 = combine(x_tr, x_tr2, y_t, y_t1)


# Pre-Processing
X = x_t2.reshape(x_t2.shape[0], 30, 90)
X = norm_subtract(X)

#np.save('Combined_Images',X)

"""
batch_size = 10
num_classes = 4
epochs = 100
# input image dimensions
img_x, img_y = 30, 90
#
X = np.load('/home/nextgen/Documents/drone_detection_models/Drone_Det_CNN/Training_Data/Combined_Images.npy')
Y = np.load('/home/nextgen/Documents/drone_detection_models/Drone_Det_CNN/Training_Data/Combined_Labels.npy')

# np.savetxt('Combined_Images.csv', X, fmt='%.2f', delimiter=',')
#Splitting the data set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Delete X,Y for RAM cleanup
del X,Y
gc.collect()

x_train = np.expand_dims(x_train,axis = -1) 
x_test = np.expand_dims(x_test,axis = -1) 
input_shape = (img_x, img_y, 1)


# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
##Convert labels to categorical 
#y_train = keras.utils.to_categorical(y_train, num_classes=4)
#y_test = keras.utils.to_categorical(y_test, num_classes=4)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

"*****************************************************************************"
model = Sequential()
model.add(Conv2D(32, (5,5),
                 padding = 'same',
                 input_shape=input_shape))
convout1 = Activation('relu')
#model.add(BatchNormalization())
model.add(convout1)
model.add(MaxPooling2D(pool_size=(2, 2)))
"*****************************************************************************"
model.add(Conv2D(32, (5,5),
                 padding = 'same',
                 input_shape=input_shape))
convout2 = Activation('relu')
#model.add(BatchNormalization())
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
"*****************************************************************************"
model.add(Conv2D(64, (5,5),
                 padding = 'same',
                 input_shape=input_shape))
convout3 = Activation('relu')
#model.add(BatchNormalization())
model.add(convout3)
model.add(Dropout(0.50))
model.add(MaxPooling2D(pool_size=(2, 2)))
"*****************************************************************************"
model.add(Conv2D(64, (5,5),
                 padding = 'same',
                 input_shape=input_shape))
convout5 = Activation('relu')
model.add(BatchNormalization())
model.add(convout5)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
"*****************************************************************************"
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
"*****************************************************************************"
# Optimizer options
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# Train with GPU's
#parallel_model = multi_gpu_model(model, gpus=8)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"*****************************************************************************"
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
"*****************************************************************************"
#Save training data to csv file
filename='model_train_CNN_1.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
callbacks_list = [csv_log]
hist = model.fit(x_train, y_train, batch_size = batch_size,verbose=1, epochs = epochs, validation_data=(x_test, y_test),callbacks=callbacks_list)
"*****************************************************************************"
# save model
model.save('SDC_Saved_Weights.h5')

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs)

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

# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

test_rate = np.mean(y_pred == y_test)
print ('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

# #Load model and continue training
# new_model = load_model('Saved_Weights.h5')
# del parallel_model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# parallel_model = multi_gpu_model(new_model, gpus=8)
# parallel_model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# hist = parallel_model.fit(x_train, y_train, batch_size = batch_size,verbose=1, epochs = epochs, validation_data=(x_test, y_test),callbacks=callbacks_list)