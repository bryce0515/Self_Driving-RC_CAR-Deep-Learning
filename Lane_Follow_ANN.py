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
#from keras.layers import LSTM, Dense
from keras.layers import Dense

class AutoPilot(object):
    def __init__(self):
        # Set up command server and wait for a connect
        print ('Waiting for Command Client')
        self.command_server_soc = socket.socket()
        self.command_server_soc.bind(('192.168.43.117', 8000))
        self.command_server_soc.listen(0) # wait for client connection
        self.command_client = self.command_server_soc.accept()[0] # wait for client to connect

        print ("command client connected!")

        print('Waiting for Video Stream')

        # create socket
        self.video_server_soc = socket.socket()
        self.video_server_soc.bind(('192.168.43.117', 8001))
        self.video_server_soc.listen(0)

        # Accept a single connection and make a file-like object out of it
        self.video_client = self.video_server_soc.accept()[0].makefile('rb')
        print("Video client connected!")
        self.send_inst = True
        self.NeuralNetwork()
        pygame.init()

    def NeuralNetwork(self):
        self.classifier = Sequential()
        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 38400))
        # Adding the second hidden layer
        self.classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
        # Adding the output  layer, 
        self.classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
        self.classifier.load_weights('Saved_Weights_ANN.h5')
        print(self.classifier)
        self.handle()

    def predict(self, samples):

        global prediction
        ret, resp = self.classifier.predict(samples)
        return resp.argmax(-1)
    
    
    def steer(self, prediction):
        global byte
        byte = b''
        prediction = prediction.argmax()
        if prediction == 2:
            byte = (bytes([1]))
            self.command_client.send(byte)
            print("Forward")
        elif prediction == 0:
            byte = (bytes([7]))
            self.command_client.send(byte)
            print("Foward Left")
        elif prediction == 1:
            byte = (bytes([6]))
            self.command_client.send(byte)
            print("Foward Right")
        elif prediction == 3:
            byte = (bytes([2]))
            print("Reverse")
            self.command_client.send(byte)
        else:
            self.stop()
        


    def stop(self):
        byte = (bytes([0]))
        self.command_client.send(byte)

    def handle(self):
        global prediction
        global byte

        # stream video frames one by one
        try:
            len_stream_image = 0
            stream_image = 0
            while self.send_inst:
                pygame.display.set_mode((100,100))
                image_len = struct.unpack('<L', self.video_client.read(struct.calcsize('<L')))[0] 
                if not image_len:
                    print ("Break as image length is null")
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.video_client.read(image_len))
                # Rewind the stream, open it as an image with PIL
                image_stream.seek(0)
                stream_image = Image.open(image_stream)
                stream_image = stream_image.convert('L') #grey scale
                # Convert image to numpy array
                stream_image = array(stream_image)
                # Feature Scaling
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                stream_image = sc.fit_transform(stream_image)
                # Check to see if full image has been loaded - this prevents errors due to images lost over network
                # Feature Scaling
                len_stream_image = stream_image.size
                # select lower half of the image
                roi = stream_image[120:340, :]
                cv2.imshow('image', stream_image)
                #cv2.imshow('mlp_image', half_gray)
                # reshape image
                image_array = roi.reshape(1, 38400).astype(np.float32)
                # neural network makes prediction
                prediction = self.classifier.predict(image_array)
                self.steer(prediction)
                print(prediction)
                self.command_client.send(byte)
                sleep(0.1)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key_input = pygame.key.get_pressed()
                         # complex orders
                        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                            print("Forward Right")
                            byte = (bytes([6]))

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            print("Forward Left")
                            byte = (bytes([7]))

                        elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                            print("Reverse Right")
                            byte = (bytes([8]))

                        elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                            print("Reverse Left")
                            byte = (bytes([9]))

                        # simple orders
                        elif key_input[pygame.K_UP]:
                            print("Forward")
                            byte = (bytes([1]))

                        elif key_input[pygame.K_DOWN]:
                            print("Reverse")
                            byte = (bytes([2]))

                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            byte = (bytes([3]))

                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            byte = (bytes([4]))

                        elif key_input[pygame.K_RSHIFT]:
                            print ('pause')
                            byte = (bytes([0]))
                            self.command_client.send(byte)
                            break
                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print ('exit')
                            self.send_inst = False
                            byte = (bytes([0]))
                            self.command_client.send(byte)
                            break
                        # send control command
                        self.command_client.send(byte)
                        
                    elif event.type == pygame.KEYUP:
                        byte = (bytes([0]))
                        # print("KEY UP")
                        # self.command_client.send(byte)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.send_inst = False
                    self.predict.stop()
                    break
               # cv2.destroyAllWindows()
 
        finally:
            self.video_client.close()
            self.video_server_soc.close()
            self.command_server_soc.close()
            self.command_client.close()
            cv2.destroyAllWindows()
            pygame.quit()


if __name__ == '__main__':
    AutoPilot()