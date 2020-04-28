import cv2
import time
import yaml
import argparse
import requests
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from make_queue import Q
from Action_Detection.pose_estimate import estimator_pose, recogn
from Action_Detection.load_path_model import load_model_path, load_pretrain_model
from flask import Flask, request, abort, send_from_directory, jsonify

app = Flask(__name__)

@app.route('/action',methods=['POST'])
def action_detect():
    if request.method == 'POST':

        filess = request.files['file']
        file_name = filess.filename[:-4]
        userid = file_name[-6:]
        filess.save('temp_action.jpg')
        print("GET IMAGE FILE NAME {} USERID {}".format(file_name, str(userid)))
        
        img = cv2.imread('temp_action.jpg')
        #action detection
        pose = estimate.get_human_info(img)
        list_result = r.action_recognition(pose)

        if len(list_result) != 0:
            for result in list_result:
                
                if result[0] == 'stand' or 'walk':
                    file_name_to_next = 'face'+ str(userid) +'.jpg'
                    cv2.imwrite(file_name_to_next, result[1])
                    print('file_name_to_next is {}'.format(file_name_to_next))
                    
                    myfiles = {'file': open(file_name_to_next ,'rb')}
                    response = requests.post(url_config['queue_url'], files = myfiles)   #send to go_queue

                elif result[0] == 'cannot detect':
                    file_name_to_next = 'cannot'+ str(userid) +'.jpg'
                    cv2.imwrite(file_name_to_next, result[1])
                    
                    myfiles = {'file': open(file_name_to_next ,'rb')}
                    response = requests.post(url_config['queue_url'], files = myfiles)
                    
                else:
                    file_name_to_next = 'not'+ str(userid) +'.jpg'
                    cv2.imwrite(file_name_to_next, img_after_action)
                    myfiles = {'file': open(file_name_to_next ,'rb')}
                    response = requests.post(url_config['queue_url'], files = myfiles)   #send to go_queue

            return 'OK'

if __name__ == "__main__":
    url_config = {}
    url_config = yaml.load(open('url_config.yml'))
    # load model from class something
    estimate = estimator_pose(load_pretrain_model('VGG_origin'))
    print('Graph path is {}'.format(load_pretrain_model('VGG_origin')))
    r = recogn("Action_Detection/Action/framewise_recognition.h5")
    
    app.run(port=5001)  



