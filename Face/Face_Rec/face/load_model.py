import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
import time
from keras.layers.normalization import BatchNormalization

def load_model_me():
    model = load_model(os.path.join('Face_Rec/face/modelfaceregV4.h5'))
    model.load_weights(os.path.join('Face_Rec/face/weightfaceregV4.h5'))

                
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    
    return model