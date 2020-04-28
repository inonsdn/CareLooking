# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import cv2 as cv
import threading
import requests
import base64
from PIL import Image
from pathlib import Path
from Action_Detection.Tracking.deep_sort import preprocessing
from Action_Detection.Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Action_Detection.Tracking.deep_sort.detection import Detection
from Action_Detection.Tracking import generate_dets as gdet
from Action_Detection.Tracking.deep_sort.tracker import Tracker
from Action_Detection.Action.action_enum import Actions
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

# 定义基本参数
file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# 初始化deep_sort
model_filename = str(file_path/'Action_Detection/Tracking/graph_model/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box颜色
trk_clr = (0, 255, 0)

def load_action_premodel(model):
    return load_model(model)

def framewise_recognize(pose, pretrained_model):
    
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])

    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        # 进行非极大抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 调用tracker并实时更新
        tracker.predict()
        tracker.update(detections)

        # 记录track的结果，包括bounding boxes及其ID
        trk_result = []
        for trk in tracker.tracks:
            # if not trk.is_confirmed() or trk.time_since_update > 1:
            #     continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
            # 标注track_ID
            trk_id = 'ID-' + str(trk.track_id)
            #cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
                # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                # 若当前帧无human，默认j=0（无效）
                j = 0

            # 进行动作分类
            if joints_norm_per_frame.size > 0:
                joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
                init_label = Actions(pred).name
                #cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
                #crop_img = frame.crop(cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2))
                crop_frame = pose[0][(ymin - 30):ymax, (xmin - 10):(xmax + 10)]
                
                #cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
                # 异常预警(under scene)
                print(init_label)
                if init_label == 'stand':
                    #b64string = base64.b64encode(crop_frame.read())
                    cv.imwrite('con_to_face.jpg',crop_frame)
                    # r = requests.post("https://2493bf19.ngrok.io/face", data=b64string)
                    print("success")    
                    
                    return frame, "finish"
                    #predict_img(crop_frame)
                    #mp.Process(target=predict_img, args=(crop_frame)).start()
                    #cv.imshow('crop',crop_frame)
                #    cv.putText(frame, 'WARNING: someone is falling down!', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                #               1.5, (0, 0, 255), 4)
            # 画track_box
            #cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
    return frame, "not finish"
