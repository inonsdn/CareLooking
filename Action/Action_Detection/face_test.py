# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:22:16 2019

@author: iNon
"""
import numpy as np
import cv2 as cv
from pathlib import Path
from PIL import Image
import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image


def detect_face(img):
    image_ori = cv.imread(img)
    #img_pred = Image.fromarray(img, 'RGB')
    for r, d, files in os.walk('test_result/'):
        i = len(files)
    print("face ",i)
    cv.imshow("test",image_ori)
    #cv.imwrite('test_result/predictimg'+str(i)+'.jpg',img)
    #img_p = cv.imread('test_result/predictimg'+str(i)+'.jpg')

    #face_cascade_path = "haarcascade_frontalface_alt.xml"
    face_cascade = cv.CascadeClassifier("Action/haarcascade_frontalface_alt.xml")
    
    
    classes = ['yok','non','ploy','unknown']
    nb_classes = len(classes)
            
    img_height, img_width = 64,64
    channels = 3
    
    #model.load_weights(os.path.join(result_dir, 'weightfacereg.h5'))
    
    img_gray = cv.cvtColor(image_ori, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    color = (0, 255, 0) 
    faces = face_cascade.detectMultiScale(img_gray) 
    print(faces)
    if faces is None:
        return print("someone stand")
    for (x,y,w,h) in faces:
        cv.rectangle(image_ori,(x,y),(x+w,y+h),color,2)
        roi_color = image_ori[y:y+h, x:x+w]
        face_img = cv.resize(roi_color,(img_height,img_width))
        cv.imwrite('test_result/face'+str(i)+'.jpg',face_img)
    
def predict_img(img):
    
    result_dir = 'Action/face/'
    image_ori = cv.imread(img)
    #img_pred = Image.fromarray(img, 'RGB')
    for r, d, files in os.walk('test_result/'):
        i = len(files)
    print("face ",i)
    cv.imshow("test",image_ori)
    #cv.imwrite('test_result/predictimg'+str(i)+'.jpg',img)
    #img_p = cv.imread('test_result/predictimg'+str(i)+'.jpg')

    #face_cascade_path = "haarcascade_frontalface_alt.xml"
    face_cascade = cv.CascadeClassifier("Action/haarcascade_frontalface_alt.xml")
    
    
    classes = ['yok','non','ploy','unknown']
    nb_classes = len(classes)
            
    img_height, img_width = 64,64
    channels = 3
    
    model = load_model(os.path.join(result_dir, 'modelfacereg.h5'))
    #model.load_weights(os.path.join(result_dir, 'weightfacereg.h5'))
    
    img_gray = cv.cvtColor(image_ori, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    color = (0, 255, 0) 
    faces = face_cascade.detectMultiScale(img_gray) 
    print(faces)
    if faces is None:
        return print("someone stand")
    for (x,y,w,h) in faces:
        cv.rectangle(image_ori,(x,y),(x+w,y+h),color,2)
        roi_color = image_ori[y:y+h, x:x+w]
        face_img = cv.resize(roi_color,(img_height,img_width))
        cv.imwrite('test_result/face'+str(i)+'.jpg',face_img)
    
    
    pred_img = cv.imread('test_result/face'+str(i)+'.jpg')
    x = image.img_to_array(pred_img)
    x = np.expand_dims(x, axis=0)
    #x = x / 255.0

    pred = model.predict(x)[0]

    top = 5
    top_indices = pred.argsort()[-top:][::-1]
    result = [(pred[i]*100, classes[i] ) for  i in top_indices]

    #result = dict(result)

    for x in result:
        print(x)
    print(pred)

predict_img("Action/training/testing/test10.jpg")