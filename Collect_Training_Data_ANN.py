
"""
Created on Mon Nov 27 12:37:06 2017

@author: bsimmons
"""

import numpy as np
from numpy import array
import cv2
import cv2 as cv
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


class CollectTrainingData_ANN(object):
    def __init__(self):
        # Set up command server and wait for a connect
        print ('Waiting for Command Client')
        self.command_server_soc = socket.socket()
        self.command_server_soc.bind(('192.168.43.117', 8000))
#        self.command_server_soc.bind(('10.0.0.1', 8000)) 
        self.command_server_soc.listen(0) # wait for client connection
        self.command_client = self.command_server_soc.accept()[0] # wait for client to connect

        print ("command client connected!")

        print('Waiting for Video Stream')

        # create socket
        self.video_server_soc = socket.socket()
 #       self.command_server_soc.bind(('10.0.0.1', 8001)) 
        self.video_server_soc.bind(('192.168.43.117', 8001))
        self.video_server_soc.listen(0)

        # Accept a single connection and make a file-like object out of it
        self.video_client = self.video_server_soc.accept()[0].makefile('rb')
        print("Video client connected!")
        self.send_inst = True

        #return self.command_client, self.video_client

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
                self.k[i, i] = 1
                self.temp_label = np.zeros((1, 4), 'float')

        pygame.init()
        self.collect_image()

    def collect_image(self,):
        saved_frame = 0
        total_frame = 0
        
        # collect images for training
        print ('Start collecting images...')

        """cv2.getTickCount function returns the number of clock-cycles after a
        reference event (like the moment machine was switched ON) to the moment this
        function is called. So if you call it before and after the function execution,
        you get number of clock-cycles used to execute a functio"""
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 54000))
        label_array = np.zeros((1, 4), 'float')

        # stream video frames one by one
        try:
            frame = 1
            #len_stream_image = 0
            stream_image = 0
            while self.send_inst:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L',  self.video_client.read(struct.calcsize('<L')))[0]
                if not image_len:
                    print ("Break as image length is null")
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write( self.video_client.read(image_len))
                # Rewind the stream, open it as an image with PIL
                image_stream.seek(0)
                stream_image = Image.open(image_stream)
                stream_image = stream_image.convert('L') #grey scale
                # Convert image to numpy array
                stream_image = array(stream_image)
                # Convert to RGB from BRG
#                from sklearn.preprocessing import StandardScaler
#                sc = StandardScaler()
#                stream_image = sc.fit_transform(stream_image)
                # Check to see if full image has been loaded - this prevents errors due to images lost over network
                #len_stream_image = stream_image.size
                # select lower half of the image
                roi = stream_image[120:450, :]
                # save streamed images
                cv2.imwrite('training_images_ANN/frame{:>05}.jpg'.format(frame), stream_image)
                #cv2.imshow('roi_image', roi)
                cv2.imshow('image', stream_image)
                # reshape the roi image into one row array
                temp_array = roi.reshape(1, 54000).astype(np.float32)
                frame += 1
                total_frame += 1
                pygame.display.set_mode((100,100))
                    # get input from human driver
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key_input = pygame.key.get_pressed()
                        # print("KEY DOWN")
                        # send command to raspberry pi (client) over socket
                        
                        # complex orders
                        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                            print("Forward Right")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            byte = (bytes([6]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            print("Forward Left")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
                            byte = (bytes([7]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                            print("Reverse Right")
                            #image_array = np.vstack((image_array, temp_array))
                            #label_array = np.vstack((label_array, self.k[4]))
                            #saved_frame += 1
                            byte = (bytes([8]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                            print("Reverse Left")
                            #image_array = np.vstack((image_array, temp_array))
                            #label_array = np.vstack((label_array, self.k[5]))
                            #saved_frame += 1
                            byte = (bytes([9]))
                            self.command_client.send(byte)

                        # simple orders
                        elif key_input[pygame.K_UP]:
                            print("Forward")
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[2]))
                            byte = (bytes([1]))
                            self.command_client.send(byte)
                            

                        elif key_input[pygame.K_DOWN]:
                            print("Reverse")
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[3]))
                            byte = (bytes([2]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            byte = (bytes([3]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
                            byte = (bytes([4]))
                            self.command_client.send(byte)

                        elif key_input[pygame.K_RSHIFT]:
                            print ('pause')
                            byte = (bytes([0]))
                            self.command_client.send(byte)
                            #break
                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print ('exit')
                            self.send_inst = False
                            byte = (bytes([0]))
                            self.command_client.send(byte)
                            break
                        # send control command
                        #self.command_client.send(byte)
                        #print(byte)
                        
                    elif event.type == pygame.KEYUP:
                        byte = (bytes([0]))
                        # print("KEY UP")
                        #self.command_client.send(byte)


#            train_images = []
            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]
#            train_images = [train,train_labels]
#            print(train_images)

            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data_ANN"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print ('Streaming duration:', time0)

            print(train.shape)
            print(train_labels.shape)
            print( 'Total frame:', total_frame)
            print ('Saved frame:', saved_frame)
            print ('Dropped frame', total_frame - saved_frame)

        finally:
            self.video_client.close()
            self.video_server_soc.close()
            self.command_server_soc.close()
            self.command_client.close()
            cv2.destroyAllWindows()
            pygame.quit()
            # sys.exit(0)

if __name__ == '__main__':
    CollectTrainingData_ANN()
