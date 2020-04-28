import os
from pathlib import Path


file_path = Path.cwd()

def load_pretrain_model(model):
    dyn_graph_path = {
        'VGG_origin': "Action_Detection/Pose/graph_models/VGG_origin/graph_opt.pb",
        'mobilenet_thin': "Action_Detection/Pose/graph_models/mobilenet_thin/graph_opt.pb"
    }
    graph_path = dyn_graph_path[model]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)

    return graph_path
    
def load_model_path(model):
    dyn_model_path = {
        'framewise': "Action_Detection/Action/framewise_recognition.h5",
        'framewise_under_scene': "Action_Detection/Action/framewise_recognition_under_scene.h5"
    }
    model_path = dyn_model_path[model]
    if not os.path.isfile(model_path):
        raise Exception('Model file doesn\'t exist, path=%s' % model_path)

    return model_path
    
