# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:16:00 2018

@author: iNonie
"""

import os
import io
import sys
import time
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
from Face_Rec.face.load_model import load_model_me
from flask import Flask, request, abort, send_from_directory
import paho.mqtt.client as paho

def on_publish(client, userdata, mid):
    print("mid: "+str(mid))

class FaceRecognition:
    def __init__(self):
        self.face_cascade_path = "Face_Rec/face/haarcascade_frontalface_alt.xml"
        global face_cascade
        global model
        self.SetGpuLimitation(0.3)
        model = load_model_me()
        face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

        
    def SetGpuLimitation(self, gpu_factor_percent):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_factor_percent
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
        print("--- Set GPU Limitation by {} ---".format(str(gpu_factor_percent)))
        
        
    def predict_img(self, img):
        
        try:
            classes = ['non','ploy','Unknown','yok']
            nb_classes = len(classes)
                    
            img_height, img_width = 64,64
            channels = 3
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
            color = (0, 255, 0) 
            faces = face_cascade.detectMultiScale(img_gray) 
            print("FACES : {} and type {}".format(faces, type(faces)))
            
            if len(faces) != 0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_color = img[y:y+h, x:x+w]
                    face_img = cv.resize(roi_color,(img_height,img_width))
                    cv2.imwrite('temp_face_only_face.jpg', face_img)
                    
                load_temp_img = cv2.imread('temp_face_only_face.jpg')
                face_img_pred = cv2.resize(load_temp_img,(img_height,img_width))
        
        
                x = image.img_to_array(face_img_pred)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0
        
                pred = model.predict(x)[0]
        
                top = 5
                top_indices = pred.argsort()[-top:][::-1]
                result = [(pred[i]*100, classes[i] ) for i in top_indices]
        
                result = dict(result)
        
                for x in result:
                    print(x)
        
                t = time.time()
                print("Finish at",t)
                
                return result
            
            else:
                return 'Others'
                
        except Exception as e:
            print("ERROR : ",e)
            print("Someone standing now")
            
            return 'Others'
