# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:12:23 2018

@author: bsimmons
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:23:50 2018

@author: bsimmons
"""

import threading
from statistics import mode
from collections import Counter
import socketserver
import cv2
import numpy as np
import math
from numpy import array
import pygame
from pygame.locals import *
from threading import Thread
import socket
import time
from time import sleep
import os
import io
import struct
import sys
from PIL import Image
import keras
from keras.models import load_model
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from sklearn.preprocessing import normalize

# distance data measured by ultrasonic sensor
sensor_data = 0

class NerualNetwork(object):
    def __init__(self):
        self.classifier = Sequential()
        
    def create(self):
        num_classes = 4
        # input image dimensions
        input_shape = (120, 450, 1)
       # self.classifier = Sequential()
       # 2D Convolution with a 3 x 3 filter and 32 seperate feature maps
       # Padding is set to keep input dimension the same as output
       # stride is set to 1 
        self.classifier.add(Conv2D(32, (3,3),
                         padding = 'same',
                         activation='relu',
                         input_shape=input_shape))
        # Second 2D Convolution
        self.classifier.add(Conv2D(32, (3,3),
                         padding = 'same',
                         activation='relu',
                         input_shape=input_shape))
        # 1st Max Pooling layer, set to a block 2 x 2 
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        # Third 2D convolution
        self.classifier.add(Conv2D(64, (3,3),
                         padding = 'same',
                         activation='relu',
                         input_shape=input_shape))
        # Fourth 2D Convolution
        self.classifier.add(Conv2D(64, (3,3),
                         padding = 'same',
                         activation='relu',
                         input_shape=input_shape))
        # Second Max pooling layer
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        # Fully Connected layer MPL (ANN)
        self.classifier.add(Flatten())
        self.classifier.add(Dense(512))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(num_classes))
        # Output layer
        self.classifier.add(Activation('softmax'))
        self.classifier.load_weights('Saved_Weights_CNN_Phase1.xml')


    def predict(self, samples):

        resp = self.classifier.predict(samples)
        return resp.argmax(-1)

        

class RCControl(object):
    def __init__(self):
        # Set up command server and wait for a connect
        print ('Waiting for Command Client')
        self.command_server_soc = socket.socket()
        self.command_server_soc.bind(('192.168.43.13', 8000))
        self.command_server_soc.listen(0) # wait for client connection
        self.command_client = self.command_server_soc.accept()[0] # wait for client to connec
        print ("command client connected!")
    
    def steer(self, prediction):
        #prediction = prediction.argmax()
        #print(prediction)
        #self.byte = b''
        if prediction == 2:
            self.byte = (bytes([1]))
            print("Prediction: Forward")
        elif prediction == 0:
            self.byte = (bytes([7]))
            print("Prediction: Foward Left")
        elif prediction == 1:
            self.byte = (bytes([6]))
            print("Prediction: Foward Right")
        elif prediction == 3:
            self.byte = (bytes([2]))
            print("Prediction: Reverse")
        else:
            self.stop()
        self.command_client.send(self.byte)


        
    def stop(self):
        #global byte
        self.byte = (bytes([0]))
        self.command_client.send(self.byte)        

class ObjectDetection(object):

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0
        
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
            #print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

            # stop sign
            if width/height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return v

class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        # parameters retrieved from piCamera_Calibration.py
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay)) + 5
        if d > 0:
            cv2.putText(image, "%.1f Inch" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d

class SensorDataHandler(socketserver.BaseRequestHandler):

    data = b''
    def handle(self):
        global sensor_data
        #try:
        while True:
            self.data = self.request.recv(1024)
            #print(self.data)
            sensor_data = int.from_bytes((self.data), byteorder='big', signed=False)
            #print "{} sent:".format(self.client_address[0])
            #print (sensor_data)
        #finally:
        #print ("Connection closed on thread 2")
    
class VideoStreamHandler(socketserver.StreamRequestHandler):
        # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10
    
    
    classifier = NerualNetwork()
    classifier.create()
    obj_detection = ObjectDetection()
    rc_car = RCControl()
    
        # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('stop_sign.xml')
    d_to_camera = DistanceToCamera()
    d_stop_sign = 25

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0
            
    def handle(self):
        global sensor_data
        global count
        global num_classes
        global classifier
        sensor_data = None
        count = 0
        flag = True
        stop_flag = False
        stop_sign_active = True
        stream_bytes = bytes()
        try:
            while True:
                pygame.display.set_mode((100,100))
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    #gray = np.array(gray)
                    stream_image = gray[120:450, :]
                                        # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)

                    # distance measurement
                    if v_param1 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h1, 300, image)
                        self.d_stop_sign = d1
                        
                    cv2.imshow('image', image)
                                            # stop conditions
                    if sensor_data is not None and sensor_data < 30:
                        print("Stop, obstacle in front")
                        self.rc_car.stop()
                    
                    elif 0 < self.d_stop_sign < 25 and stop_sign_active:
                        print("Stop sign ahead")
                        self.rc_car.stop()

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print ("Stop time: %.2fs" % self.stop_time)

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False
                            
                    else:
                        count += 1
                        if count == 1 and flag == True:
                            stream_image = np.divide(stream_image,255)
                            # reshape image to size [120 x 450 x 1]
                            image_array = stream_image.reshape(1,120, 450, 1).astype(np.float32)
                            # neural network makes prediction
                            prediction = self.classifier.predict(image_array)
                            self.rc_car.steer(prediction)
                            self.stop_start = cv2.getTickCount()
                            self.d_stop_sign = 25
                        if count == 10:
                            stream_image = np.divide(stream_image,255)
                            # reshape image to size [120 x 450 x 1]
                            image_array = stream_image.reshape(1,120, 450, 1).astype(np.float32)
                            # neural network makes prediction
                            prediction = self.classifier.predict(image_array)
                            count = 0
                            flag = False
                            self.rc_car.steer(prediction)
                            self.stop_start = cv2.getTickCount()
                            self.d_stop_sign = 25
    
                            if stop_sign_active is False:
                                self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                                if self.drive_time_after_stop > 5:
                                    stop_sign_active = True
    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.send_inst = False
                        self.predict.stop()
                        break
     
        finally:
            print("Connection closed on thread 1")
            cv2.destroyAllWindows()
            pygame.quit()

class ThreadServer(object):
    
    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    distance_thread = threading.Thread(target=server_thread2, args=('192.168.43.13', 8002))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread('192.168.43.13', 8001))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()