import cv2
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

        # if got the image file
        if filess != None:
            userid = filess.filename[-10:-4]       # extract userid from filename
            
            print("GET IMAGE FILE NAME {} USERID {}".format(filess.filename, str(userid)))

            # check if got image file for action process
            if filess.filename[0] == 'a':
                queue.time_process('start')
                queue.save_for(userid, filess, 'action')
                
                status_action_process = queue.get_status('action')
                
                # check the action queue are ready to process or not
                # if ready, send the next image in queue to action api
                if status_action_process == 'ready':
                    queue.update_status('action', 'busy')
                    status = queue.thread('action', action_url)

            # check if got image file for face process
            elif filess.filename[0] == 'f':
                queue.save_for(userid, filess, 'face')
                
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                # check action queue have or not
                # if not, change action status to 'ready'
                if action_queue == False:
                    queue.update_status('action','ready')
                
                # if have queue, send the next image
                else:
                    status = queue.thread('action', action_url)

                # check the face queue are ready to process or not
                # if ready, send the next image in queue to face api
                if status_face_process == 'ready':
                    queue.update_status('face', 'busy')
                    status = queue.thread('face', face_url)

            # check if got not image file
            elif filess.filename[0] == 'n':
                queue.save_for(userid, filess, 'not')
                
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                if action_queue == False:
                    queue.update_status('action','ready')
                
                else:
                    status = queue.thread('action', action_url)
            
            # check if got cannot image file
            elif filess.filename[0] == 'c':
                status_action_process = queue.get_status('action')
                status_face_process = queue.get_status('face')
                action_queue = queue.check_queue('action')
                
                if action_queue == False:
                    queue.update_status('action','ready')
                
                
                else:
                    status = queue.thread('action', action_url)
                    
                    
        # if got the text
        elif status_img != None:
            face_queue = queue.check_queue('face')

            # check face queue have or not
            # if not, change face status to 'ready'
            if face_queue == False:
                queue.update_status('face','ready')
                
                queue.time_process('end')
                print(queue.time_process(calculate=True))

            # if have queue, send the next image
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
    