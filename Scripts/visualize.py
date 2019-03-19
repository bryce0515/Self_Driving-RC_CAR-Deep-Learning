# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:16:25 2018

@author: mlearn
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import random
#from Train_DenseNet import DenseNet
from keras.models import model_from_json
from keras import activations
from keras.models import load_model
#import keras
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad,Adadelta, Adamax,Nadam
from keras.models import Sequential,load_model
from keras.layers import Lambda, concatenate

from vis.visualization import visualize_activation
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam
import matplotlib.cm as cm
#from vis.visualization import get_num_filters

#Load CNN weights
model = load_model("C:/Users/bsimmons/Documents/self_driving_car/Saved_Weights/SDC_Saved_Weights.h5")

#Load Training Data
input_img = np.load('C:/Users/bsimmons/Documents/self_driving_car/Training_Data/Combined_Images.npy')

# ("Rows,columns,channels")
#input_shape = (30, 90, 1)
img1 = np.expand_dims(input_img,3)

# ("tensor,Rows,columns,channels")
#input_shape = (1,30, 90, 1)
test_images = np.expand_dims(img1,0)
print(np.shape(test_images))

#Print out model summary
model.summary()

#"""
#
#Choose random value in from the training set and classify
#input_image = 2D numpy array
#pred = prediction from CNN
#
#"""
leftc = 0
rightc = 0
fowardc = 0
reversec = 0
#
save_le = []
save_ri = []
save_fo = []
save_re = []
#
import random
amount = len(test_images[0])
for x in range(amount):
    #Lets time the CNN prediction
    #image = random.choice(np.arange(amount))
    #print(x)
    #start = time.time()
    #Send + recieve from CNN (x0,x1,x2,x3) 
    pred = model.predict(test_images[:,x])
    #Calculate largest probability 
    preda = np.argmax(pred)
    #Print time it took to predict + argmax
    #print("\n Duration to retrieve a prediction from CNN is = {0} ms\n".format(round(time.time() - start,6)/1e-3))
#    
#    #Print out action results
#    #print(np.round(pred[0],4))
#    #results = np.round(pred[0],4)
    if preda == 0:
        leftc+=1
        save_le.append(test_images[:,x])
        #print(" \n Prediction = left")
    elif preda == 1:
        rightc+=1
        save_ri.append(test_images[:,x])
        #print("\n Prediction = right")
    elif preda == 2:
        fowardc+=1
        save_fo.append(test_images[:,x])
        #print("\n Prediction = foward")
    else:
        reversec+=1
        save_re.append(test_images[:,x])
print("\n ***** total actions ***** \n\n  left = {0}, right = {1}, foward = {2}, reverse = {3}\n\n".format(leftc,rightc,fowardc,reversec))
        #print("\n Prediction = reverse")
#    plt.imshow(input_img[x])
#    plt.pause(.01)
        
left,right,foward,reverse = [np.array(save_le),np.array(save_ri),np.array(save_fo),np.array(save_re)]
#
##show images from each category
#print("*** Left ***")
plt.imshow(left[2][0][:,:,0])
np.save("Left_Image_Sample",left[2][0][:,:,0])
plt.pause(.01)
#
#print("*** Right ***")
plt.imshow(right[0][0][:,:,0])
np.save("Right_Image_Sample",right[2][0][:,:,0])
plt.pause(.01)
#
#print("*** Foward ***")
plt.imshow(foward[0][0][:,:,0])
np.save("Foward_Image_Sample",foward[2][0][:,:,0])
plt.pause(.01)
#
#print("*** Reverse ***")
for x in range(25,100):
    print(x)
    plt.imshow(reverse[92][0][:,:,0])
    np.save("Reverse_Image_Sample",reverse[92][0][:,:,0])
    plt.pause(.01)
                
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

def norm_images(image):
    imag = []
    for x in range(len(image)):
        x_t = norm_image(image[x])
        imag.append(x_t)
    return np.array(imag)

from skimage.transform import resize
tmp = []
for x in range(len(images)):
    r = resize(foward_image[0], (30, 90),mode='constant')
    s = subtract_median(r)
    n = norm_image(s)
    tmp.append(n)
histogram = plt.hist(n.ravel(), bins=256, range=(0.0, 1), fc='k', ec='k')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Gray Scale Image')
plt.show()
"""
Load sample Images from each classification
*** left,right,fowar,reverse ***

"""
model.summary()

left_sample = np.load("Left_Image_Sample.npy")
right_sample = np.load("Right_Image_Sample.npy")
foward_sample = np.load("Foward_Image_Sample.npy")
reverse_sample = np.load("Reverse_Image_Sample.npy")

print("\n Input Image ")
plt.imshow(left_sample)
plt.pause(.01)
plt.imshow(right_sample)
plt.pause(.01)
plt.imshow(foward_sample)
plt.pause(.01)
plt.imshow(reverse_sample)
plt.pause(.01)


layer_idx = utils.find_layer_idx(model, 'activation_8')
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
img = visualize_activation(model, layer_idx, filter_indices=1)
plt.imshow(img[:,:,0])

layer_idx = utils.find_layer_idx(model, 'conv2d_4')

#for modifier in ['guided', 'relu']:
#    for x in range(2):
##        for i, img in enumerate([heat_map_img, heat_map_img]):    
#            # 20 is the imagenet index corresponding to `ouzel`
temp = foward_sample #*255
# ("Rows,columns,channels")
#input_shape = (30, 90, 1)
temp = np.expand_dims(temp,3)

# ("tensor,Rows,columns,channels")
#input_shape = (1,30, 90, 1)
new_image = np.expand_dims(temp,0)
grads_plus = np.zeros((30,90))
print(np.shape(new_image))
for x in range(32):
    grads = visualize_saliency(model, layer_idx, filter_indices=x, seed_input=new_image,backprop_modifier='guided')
    grads_plus+=np.array(grads)
    #combined.append(grads)
#    grads1= visualize_cam(model, layer_idx, filter_indices=x,seed_input=new_image,backprop_modifier='guided') 
#    jet_heatmap = np.uint8(cm.jet(grads1)[..., :3]*255 )
#    jet_heatmap = jet_heatmap[:,:,0]
#    jet_heatmap = subtract_median(jet_heatmap)
#    jh = norm_image(jet_heatmap)
    #new_img = np.copy(jh[:,:,0])
    print(x)
#    plt.imshow(overlay(jh, reverse_sample*0.25))
#    plt.pause(.01)
    
    plt.imshow(overlay(grads_plus, foward_sample*5))
    plt.pause(.01)


