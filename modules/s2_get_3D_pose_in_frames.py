import os
import sys
import yaml
import cv2
from utils.commons import read_yaml

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(ROOT + '/mmpose')

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

# -----------------------------

# For Run Once
INFERENCER_CONFIG  = {'pose2d': None, 'pose2d_weights': None, 'scope': 'mmpose', 'device': None, 'det_model': None, 'det_weights': None, 'det_cat_ids': 0, 'pose3d': 'human3d', 'pose3d_weights': None}
ACTION_PATH = 'data/SequentialFames/'
JSON_OUTPUT_PATH = 'data/3dSkeleton/'

inferencer = MMPoseInferencer(**INFERENCER_CONFIG)

for frame_folder in os.listdir(ACTION_PATH):
    for image in os.listdir(os.path.join(ACTION_PATH,frame_folder)):
        image_path = os.path.join(ACTION_PATH,frame_folder,image)
        output_path = os.path.join(JSON_OUTPUT_PATH,frame_folder,image.split('.')[0])

        inferencer_input = {'inputs': image_path,'show':False,'draw_bbox': False, 'draw_heatmap': True, 'bbox_thr': 0.3, 'nms_thr': 0.3, 'kpt_thr': 0.3, 'tracking_thr': 0.3, 'use_oks_tracking': False, 'norm_pose_2d': False, 'rebase_keypoint_height': False, 'radius': 3, 'thickness': 1, 'skeleton_style': 'mmpose', 'black_background': False, 'vis_out_dir': '', 'pred_out_dir': output_path}
        for _ in inferencer(**inferencer_input):
            pass

# --------------------------------------
# For Small Runs
INFERENCER_CONFIG  = {'pose2d': None, 'pose2d_weights': None, 'scope': 'mmpose', 'device': None, 'det_model': None, 'det_weights': None, 'det_cat_ids': 0, 'pose3d': 'human3d', 'pose3d_weights': None}
ACTION_PATH = 'data/SequentialFames/Action1' # for each action
JSON_OUTPUT_PATH = 'data/3dSkeleton/Action1' # for each action

inferencer = MMPoseInferencer(**INFERENCER_CONFIG)


for image in os.listdir(ACTION_PATH):
    image_path = os.path.join(ACTION_PATH,image)
    output_path = os.path.join(JSON_OUTPUT_PATH,image.split('.')[0])

    inferencer_input = {'inputs': image_path,'show':False,'draw_bbox': False, 'draw_heatmap': True, 'bbox_thr': 0.3, 'nms_thr': 0.3, 'kpt_thr': 0.3, 'tracking_thr': 0.3, 'use_oks_tracking': False, 'norm_pose_2d': False, 'rebase_keypoint_height': False, 'radius': 3, 'thickness': 1, 'skeleton_style': 'mmpose', 'black_background': False, 'vis_out_dir': '', 'pred_out_dir': output_path}
    for _ in inferencer(**inferencer_input):
        pass

