# make class function
# 1. load pretraied model
# 2. draw pose
# 3. recognition
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from Action_Detection.load_path_model import load_pretrain_model
from Action_Detection.Pose.coco_format import CocoPart, CocoColors, CocoPairsRender
from Action_Detection.utils import load_pretrain_model
from Action_Detection.Pose.pose_estimator import estimate

from pathlib import Path
from Action_Detection.Tracking.deep_sort import preprocessing
from Action_Detection.Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Action_Detection.Tracking.deep_sort.detection import Detection
from Action_Detection.Tracking import generate_dets as gdet
from Action_Detection.Tracking.deep_sort.tracker import Tracker
from Action_Detection.Action.action_enum import Actions
from keras.models import load_model
from pathlib import Path
import yaml 

file_path = Path.cwd()

def SetGpuLimitation(gpu_factor_percent):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_factor_percent
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    print("--- Set GPU Limitation by {} ---".format(str(gpu_factor_percent)))

def load_action_premodel(model):
    return load_model(model)
    
def keep_temp_variable(variables, filename):
    with open(filename + '.yaml', 'w') as file:
        documents = yaml.dump(variables, file)
    
def read_temp_variable(filename):
    with open(filename + '.yaml', 'r') as file:
        variables = yaml.load(file)
    
    return variables
    
class estimator_pose:
    
    Thickness_ratio = 2
    
    def __init__(self, graph_path, target_size=(368, 368)):
        self.SetGpuLimitation(0.5)
        self.target_size = target_size
        # load graph
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.compat.v1.Session(graph=self.graph)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.heatMat = self.pafMat = None
        
        
    def SetGpuLimitation(self, gpu_factor_percent):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_factor_percent
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
        print("--- Set GPU Limitation by {} ---".format(str(gpu_factor_percent)))
        
    
    def get_human_info(self, npimg, imgcopy=False):
        if npimg is None:
            raise Exception('The frame does not exist.')

        rois = []
        infos = []
        # _get_scaled_img
        if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
            # resize
            npimg = cv.resize(npimg, self.target_size)
            rois.extend([npimg])
            infos.extend([(0.0, 0.0, 1.0, 1.0)])

        output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})

        heat_mats = output[:, :, :, :19]
        paf_mats = output[:, :, :, 19:]

        output_h, output_w = output.shape[1:3]
        max_ratio_w = max_ratio_h = 10000.0
        for info in infos:
            max_ratio_w = min(max_ratio_w, info[2])
            max_ratio_h = min(max_ratio_h, info[3])
        mat_w, mat_h = int(output_w / max_ratio_w), int(output_h / max_ratio_h)

        resized_heat_mat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
        resized_paf_mat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
        resized_cnt_mat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
        resized_cnt_mat += 1e-12

        for heatMat, pafMat, info in zip(heat_mats, paf_mats, infos):
            w, h = int(info[2] * mat_w), int(info[3] * mat_h)
            heatMat = cv.resize(heatMat, (w, h))
            pafMat = cv.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)
            # add up
            resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(
                resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
            resized_paf_mat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
            resized_cnt_mat[max(0, y):y + h, max(0, x):x + w, :] += 1

        self.heatMat = resized_heat_mat
        self.pafMat = resized_paf_mat / (np.log(resized_cnt_mat) + 1)

        humans = estimate(self.heatMat, self.pafMat)
    
    # draw pose rgb
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joints, bboxes, xcenter = [], [], []

        # for record and get dataset
        joints_norm_per_frame = []

        for human in humans:
            xs, ys, centers = [], [], {}
            # วาดโหนดที่เกี่ยวข้องทั้งหมดบนรูปภาพ
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():

                    # สำหรับข้อมูลที่ขาดหายไปให้กรอก 0
                    joints_norm_per_frame += [0.0, 0.0]
                    continue

                body_part = human.body_parts[i]
                center_x = body_part.x * image_w + 0.5
                center_y = body_part.y * image_h + 0.5
                center = (int(center_x), int(center_y))
                centers[i] = center

                joints_norm_per_frame += [round(center_x/1280, 2), round(center_y/720, 2)]

                xs.append(center[0])
                ys.append(center[1])
                # วาดคะแนนร่วมกัน
                #cv.circle(npimg, center, 3, CocoColors[i], thickness=estimator_pose.Thickness_ratio * 2,
                #          lineType=8, shift=0)
                          
            # เชื่อมต่อข้อต่อที่เป็นของคนคนเดียวกันตามแต่ละส่วน
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                #cv.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order],
                #        thickness=estimator_pose.Thickness_ratio, lineType=8, shift=0)

            # สร้างพื้นที่ ROI จากข้อมูลจุดร่วมของแต่ละคน
            tl_x = min(xs)
            tl_y = min(ys)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            bboxes.append([tl_x, tl_y, width, height])

            # บันทึกโหนดที่เกี่ยวข้องทั้งหมดของแต่ละเฟรม
            joints.append(centers)

            # บันทึกจุดที่ 1 ของ coco เป็น xcenter
            if 1 in centers:
                xcenter.append(centers[1][0])

        return npimg, joints, bboxes, xcenter, joints_norm_per_frame
        
    # framewise recognition
class recogn:
    
    
    def __init__(self, action_model_path):
        # self.SetGpuLimitation(0.2)
        # global pretrained_model
        # pretrained_model = load_model(model_path)

        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        
        global graph
        global sess
        global pretrained_model
        global encoder
        graph = tf.compat.v1.get_default_graph()
        sess = tf.Session()
        set_session(sess)
        pretrained_model = load_action_premodel(action_model_path)
        
        
        # 初始化deep_sort
        # self.model_filename = str(file_path/'Action_Detection/Tracking/graph_model/mars-small128.pb')
        encoder = gdet.create_box_encoder('Action_Detection/Tracking/graph_model/mars-small128.pb', batch_size=1)
        self.metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        
        # track_box颜色
        self.trk_clr = (0, 255, 0)
        
    def SetGpuLimitation(self, gpu_factor_percent):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_factor_percent
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
        print("--- Set GPU Limitation by {} ---".format(str(gpu_factor_percent)))
        
    def action_recognition(self, pose):
        global graph
        global sess
        global pretrained_model
        global encoder
        
        all_frame = []
        
        npimg, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
        joints_norm_per_frame = np.array(pose[-1])
        print('Variable {var} : {val}'.format(var='joints', val=joints))
        print('Variable {var} : {val}'.format(var='bboxes', val=bboxes))
        print('Variable {var} : {val}'.format(var='xcenter', val=xcenter))
        print('Variable {var} : {val}'.format(var='joints_norm_per_frame', val=joints_norm_per_frame))
            
        if bboxes:
            bboxes = np.array(bboxes)
            features = encoder(npimg, bboxes)
    
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
            print('Variable {var} : {val}'.format(var='detections', val=detections))
            # 进行非极大抑制
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            print('Variable {var} : {val}'.format(var='boxes', val=boxes))
            print('Variable {var} : {val}'.format(var='scores', val=scores))
            print('Variable {var} : {val}'.format(var='indices', val=indices))
            print('Variable {var} : {val}'.format(var='detections', val=detections))
            
            
            # ติดตามการโทรและปรับปรุงในเวลาจริง
            self.tracker.predict()
            self.tracker.update(detections)
            
            print('Variable {var} : {val}'.format(var='self.tracker.update(detections)', val=self.tracker.update(detections)))
            
            # บันทึกผลลัพธ์การติดตามรวมถึงกล่องขอบเขตและรหัสของพวกเขา
            trk_result = []
            for trk in self.tracker.tracks:
                print('Variable {var} : {val}'.format(var='trk', val=trk))
                # if not trk.is_confirmed() or trk.time_since_update > 1:
                #     continue
                bbox = trk.to_tlwh()
                trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
                # ทำเครื่องหมายtrack_ID
                trk_id = 'ID-' + str(trk.track_id)
                # cv.putText(npimg, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, self.trk_clr, 3)
                
            print('Variable {var} : {val}'.format(var='trk_result', val=trk_result))
            
            
            for d in trk_result:
                xmin = int(d[0])
                ymin = int(d[1])
                xmax = int(d[2]) + xmin
                ymax = int(d[3]) + ymin
                # id = int(d[4])
                try:
                    # xcenter คือค่าพิกัด x ของจุดร่วมข้อ 1 (คอ) ของมนุษย์ในกรอบภาพ
                    # จับคู่ ID โดยการคำนวณระยะห่างระหว่าง track_box และศูนย์ xcenter ของมนุษย์
                    tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                    j = np.argmin(tmp)
                except:
                    # หากไม่มีมนุษย์อยู่ในเฟรม ปัจจุบันค่าเริ่มต้น j = 0 (ไม่ถูกต้อง)
                    j = 0
                print('Variable {var} : {val}'.format(var='j', val=j))
                # จำแนกการกระทำ
                if len(joints_norm_per_frame) > 0:
                    # with graph.as_defalut()
                    joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                    print('Variable {var} : {val}'.format(var='pretrained_model', val=pretrained_model))
                    
                    joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                    with graph.as_default():
                        set_session(sess)
                        pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
                    init_label = Actions(pred).name
                    # 显示动作类别
                    print(init_label)
                    # cv.putText(npimg, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, self.trk_clr, 3)
                    # 异常预警(under scene)
                    if init_label == 'stand':
                        print("Standing")
                        
                        
                    # 画track_box
                    # cv.rectangle(npimg, (xmin - 10, ymin - 30), (xmax + 10, ymax), self.trk_clr, 2)
                    y = ymin - 30
                    h = ymax - y
                    x = xmin - 10
                    w = xmax - x
                    crop_img = npimg[y:y+h, x:x+w]
                
                    all_frame.append((init_label, crop_img))    
                    
        return all_frame
            