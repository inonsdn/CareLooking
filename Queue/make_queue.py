import os
import io
import cv2
import time
import yaml
import base64
import numpy as np
from PIL import Image
import requests, threading
from concurrent.futures import ThreadPoolExecutor

def request_url(imgfiles, url):
    print("IN FUNC REQUEST")
    print(imgfiles[0], type(imgfiles))
    
    if imgfiles[19] == 'a':     # imgfiles = 'api_database/image/action....'
        print("IN a COND")  
        myfiles = {'file': open(imgfiles ,'rb')}
        r = requests.post(url, files=myfiles)
        return r

    elif imgfiles[19] == 'f':
        myfiles = {'file': open(imgfiles ,'rb')}
        r = requests.post(url, files=myfiles)
        return r
    
class Q(object):
    action_read_queue = 0
    face_read_queue = 0
    action_status = 'ready'
    face_status = 'ready'
    action_save_queue = 0
    face_save_queue = 0
    not_save_queue = 0
    action_queue = []
    face_queue = []
    start_time = 0
    end_time = 0

    for r, d, files in os.walk('api_database/image/'):
        for num in range(len(files)):
            # files name is 0actionCL0000.jpg # check order and serial
            if files[num][0] == 'a':  
                   action_save_queue += 1
            elif files[num][0] == 'f':  
                   face_save_queue += 1
            elif files[num][0] == 'n':  
                   not_save_queue += 1


    def __init__(self):
        self.path = 'api_database/image/'
        self.index = {}
        self.number = []
        self.counting = 0
        
        for i in range(0,10):
            self.number.append(str(i))


    def time_process(self, start_or_end='start', calculate=False):
        if calculate == False:
            if start_or_end == 'start':
                Q.start_time = time.time()
                
            elif start_or_end == 'end':
                Q.end_time = time.time()
        
        elif calculate == True:
            return 'TOTAL TIME USE : {}'.format(Q.end_time - Q.start_time)
            
            
    def get_status(self, state):
        if state == 'action':
            return Q.action_status
            
        elif state == 'face':
            return Q.face_status
    
    
    def update_status(self, state, new_status):
        if state == 'action':
            Q.action_status = new_status
            
        elif state == 'face':
            Q.face_status = new_status
    
    
    def check_queue(self, state):
        if state == 'action':
            if len(Q.action_queue) == 0:
                return False
                
            else:
                return True
                
        elif state == 'face':
            if len(Q.face_queue) == 0:
                return False
                
            else:
                return True
        
        
    def save_for(self, userid, image, for_what):
        '''
        save image for prepare to the action detection in object that is user id or serial number of device

        By checking the quantity of picture that from same serial 

        Then saving the new image to next order
        '''
        
        if for_what == 'action':
            Q.action_save_queue += 1
            file_name = self.path + for_what + str(Q.action_save_queue) +  userid + '.jpg'
            image.save(file_name)
            Q.action_queue.append(file_name)

        elif for_what == 'face':
            Q.face_save_queue += 1
            file_name = self.path + for_what + str(Q.face_save_queue) + userid + '.jpg'
            image.save(file_name)
            Q.face_queue.append(file_name)
            
        elif for_what == 'not':
            Q.face_save_queue += 1
            file_name = self.path + for_what + str(Q.not_save_queue) + userid + '.jpg'
            image.save(file_name)

    
    def read_for(self, for_what):
        if for_what == 'action':
            file_name = Q.action_queue.pop(0)
        
        elif for_what == 'face':
            file_name = Q.face_queue.pop(0)

        return file_name

  
    def rename_delete(self, userid, name_file):
        os.remove(self.path + name_file + userid + '.jpg')
        for r, d, files in os.walk(self.path):
            for num in range(len(files)):
                if files[num][-10:-4] == userid:
                    os.rename(self.path + name_file + userid + '.jpg', self.path + str(int(name_file[0])-1) + name_file[1:] + userid + '.jpg')
    

    def thread(self, state, url):
        send_url = []
        file_list = []
        
        if state == 'action':
            try:
                img_action = self.read_for('action')
            except:
                img_action = 0
                
            if type(img_action) is not int:
                file_list.append(img_action)
                send_url.append(url)

        elif state == 'face':
            try:
                img_face = self.read_for('face')
            except:
                img_face = 0
        
            if type(img_face) is not int:
                file_list.append(img_face)
                send_url.append(url)

        else:
            return 'Wrong Argument : state have only action and face'
        pool = ThreadPoolExecutor(max_workers=5)
        result = pool.map(request_url, file_list, send_url)

        return result
