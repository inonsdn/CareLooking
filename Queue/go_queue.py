import cv2 as cv
import os
import io
import sys
import argparse
import numpy as np
import time
import threading
from PIL import Image
import base64
from flask import Flask, request, abort, send_from_directory, jsonify
from make_queue import Q
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)

queue_url = os.getenv('QUEUE_URL', 'https://127.0.0.1:5000/server')
action_url = os.getenv('ACTION_URL', 'https://127.0.0.1:5001/action')
face_url = os.getenv('FACE_URL', 'https://127.0.0.1:5002/face')

@app.route("/server", methods=['POST'])
def server():
    if request.method == 'POST':

        path_save = 'api_database/image/'
        
        filess = None
        status_img = None
        
        try:
            filess = request.files['file']
        except:
            status_img = request.get_data(as_text=True)
    
        if filess != None:
            print(type(filess))                     # filename waitCL0001.jpg
            userid = filess.filename[-10:-4]       # CL0001
            
            print("GET IMAGE FILE NAME {} USERID {}".format(filess.filename, str(userid)))
            if filess.filename[0] == 'a':
                queue.time_process('start')
                queue.save_for(userid, filess, 'action')
                
                status_action_process = queue.get_status('action')
                
                if status_action_process == 'ready':
                    queue.update_status('action', 'busy')
                    status = queue.thread('action', action_url)
            
            elif filess.filename[0] == 'f':
                queue.save_for(userid, filess, 'face')
                
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                
                if action_queue == False:
                    queue.update_status('action','ready')
                
                else:
                    status = queue.thread('action', action_url)
                    
                if status_face_process == 'ready':
                    queue.update_status('face', 'busy')
                    status = queue.thread('face', face_url)
            
            elif filess.filename[0] == 'n':
                queue.save_for(userid, filess, 'not')
                
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                if action_queue == False:
                    queue.update_status('action','ready')
                
                else:
                    status = queue.thread('action', action_url)
                    
            elif filess.filename[0] == 'c':
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                if action_queue == False:
                    queue.update_status('action','ready')
                
                else:
                    status = queue.thread('action', action_url)
                    
                    
                    
        elif status_img != None:
            face_queue = queue.check_queue('face')
            if face_queue == False:
                queue.update_status('face','ready')
                
                queue.time_process('end')
                print(queue.time_process(calculate=True))
        
            else:
                status = queue.thread('face', face_url)
        print("------------------------------------------------------------------")
        print('action_read_queue is {}'.format(queue.action_read_queue))
        print('face_read_queue is {}'.format(queue.face_read_queue))
        print('action_status is {}'.format(queue.action_status))
        print('face_status is {}'.format(queue.face_status))
        print('action_save_queue is {}'.format(queue.action_save_queue))
        print('face_save_queue is {}'.format(queue.face_save_queue))
        print('not_save_queue is {}'.format(queue.not_save_queue))
        print('action_queue is {}'.format(queue.action_queue))
        print('face_queue is {}'.format(queue.face_queue))
        print("------------------------------------------------------------------")
        print("###################################################################")

        return 'OK'

if __name__ == "__main__":
    queue = Q()
    app.run(port=5000)  
    