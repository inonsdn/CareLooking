import cv2
import time
import yaml
import requests
import paho.mqtt.client as paho
from make_queue import Q
from Face_Rec.face.app_face import FaceRecognition
from flask import Flask, request, abort, send_from_directory, jsonify


app = Flask(__name__)
queue_url = os.getenv('QUEUE_URL', 'https://127.0.0.1:5000/server')

def on_publish(client, userdata, mid):
    print("mid: "+str(mid))

@app.route('/face',methods=['POST'])
def action_detect():
    if request.method == 'POST':

        filess = request.files['file']
        file_name = filess.filename[:-4]
        userid = file_name[-6:]
        filess.save('temp_face.jpg')
        print("GET IMAGE FILE NAME {} USERID {}".format(file_name, str(userid)))
        
        img = cv2.imread('temp_face.jpg')
        results = faces.predict_img(img)
        
        print(results)
        
        client.publish(data_config['topic'] + str(userid), results)
        
        response = requests.post(queue_url, data = 'done')
        
        return 'OK'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input model path')
    parser.add_argument('--model', type=str, default='Face_Rec/face/modelfaceregV4.h5')
    parser.add_argument('--weight', type=str, default='Face_Rec/face/weightfaceregV4.h5')
    args = parser.parse_args()

    data_config = {}
    data_config = yaml.load(open('config.yml'))
    client = paho.Client()
    client.on_publish = on_publish
    client.connect(data_config['ip_address'], data_config['port'])
    client.loop_start()
    faces = FaceRecognition(args.model, args.weight)
    app.run(port=5002)  