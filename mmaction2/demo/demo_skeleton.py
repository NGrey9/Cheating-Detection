# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile

import numpy as np
from math import sqrt
import cv2
import mmcv
import mmengine
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_skeleton,
                           init_recognizer, pose_inference, pose_inference_2)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
from numpy import asarray as array
from tqdm import tqdm

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def visualize(args, frames, data_samples, action_label):
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(data_samples[0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton for each frame')
    for d, f in track_iter_progress(list(zip(data_samples, frames))):
        f = mmcv.imconvert(f, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
        vis_frame = visualizer.get_image()
        cv2.putText(vis_frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        vis_frames.append(vis_frame)

    vid = mpy.ImageSequenceClip(vis_frames, fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)



def dissect_pose_results(pose_results):
    """
    Function is to get the nearest bbox by compute the distance of 2 bboxes
    when we have pre_bbox and current_bboxes
    Args: 
        pose_results: list of dict of pose_result which contains inaccurate individual coordinates
    Returns:
        new_pose_resuts: list of dict of pose_result which contains accurate coordinates
    """
    dissected_pose_results = []
    for fr in range(len(pose_results)):
        for p in range(len(pose_results[fr]['bboxes'])):
            data = {'bboxes':array([pose_results[fr]['bboxes'][p]]),
                    'bbox_scores':array([pose_results[fr]['bbox_scores'][p]]),
                    'keypoints_visible':array([pose_results[fr]['keypoints_visible'][p]]),
                    'keypoints':array([pose_results[fr]['keypoints'][p]]),
                    'keypoint_scores':array([pose_results[fr]['keypoint_scores'][p]])}
            if p == 0:
                dissected_pose_results.append([data])
            else:
                dissected_pose_results[fr].append(data)
    return dissected_pose_results

def get_distance(pre_point, curr_point):
    distance = sqrt((curr_point[0][0]-pre_point[0][0])**2 + (curr_point[0][1]-pre_point[0][1])**2)
    return distance
def match_pose(dissected_pose_results):
    matched_results = []
    list_consider = []
    for i in range(len(dissected_pose_results[0])):
        matched_results.append([dissected_pose_results[0][i]])
    for abc in range(1,len(dissected_pose_results)):
        for p in range(len(dissected_pose_results[0])):
            fr = len(matched_results[p])
            list_consider.append(dissected_pose_results[fr][p])
        for p in range(len(matched_results)):
            list_distances = []
            for i in range(len(list_consider)):
                distance = get_distance(matched_results[p][0]['bboxes'],list_consider[i]['bboxes'])
                list_distances.append(distance)
            min_id = list_distances.index(min(list_distances))
            chosen_value = list_consider.pop(min_id)
            matched_results[p].append(chosen_value)
            
        
    return matched_results
            
                
        

            


# def post_processing(pose_results):
#     """
#     Function is to extract single pose result from multiple pose results
#     Args: pose_results: multiple pose results
#     Return: pose_result_list: single pose result
#     """
#     pose_result_list = []
#     pre_j = 0
    
#     for i in range(len(pose_results)):
#         for j in range(len(pose_results[i]['bbox_scores'])):
#             bbox_scores = pose_results[i]['bbox_scores'][j],
#             keypoint_scores = pose_results[i]['keypoint_scores'][j],
#             keypoints = pose_results[i]['keypoints'][j],
#             keypoints_visible = pose_results[i]['keypoints_visible'][j],
#             bboxes = pose_results[i]['bboxes'][j]
            
#             data = {'bbox_scores':bbox_scores,
#                     'keypoint_scores':keypoint_scores,
#                     'keypoints':keypoints,
#                     'keypoints_visible':keypoints_visible,
#                     'bboxes':bboxes}
            
#             if i == 0 or j > pre_j:
#                 pose_result_list.append([data])
#             else:
#                 current_bboxes = pose_results[i]['bboxes']
                
#                 print(f"prebbox.size : {prebboxes}")
#                 print(f"curr_bboxe.size : {current_bboxes}")

#                 nearest_bbox = get_nearest_bbox(prebboxes[j],current_bboxes)
#                 pose_result_list[j].append({'bbox_scores':bbox_scores,
#                                             'keypoint_scores':keypoint_scores,
#                                             'keypoints':keypoints,
#                                             'keypoints_visible':keypoints_visible,
#                                             'bboxes':nearest_bbox})
#             prebboxes = []
#             prebboxes.append(pose_results[i]['bboxes'])
#             # else:
#             #     if 
#             #     all_poses[j].append(data)
#         pre_j = j
#     return pose_result_list

def post_processing(pose_results):
    pose_result_list = []
    for i in range(len(pose_results)):
        for j in range(len(pose_results[i]['bboxes'])):
            bbox_scores = np.asarray([pose_results[i]['bbox_scores'][j]])
            keypoint_scores = np.asarray([pose_results[i]['keypoint_scores'][j]])
            keypoints = np.asarray([pose_results[i]['keypoints'][j]])
            keypoints_visible = np.asarray([pose_results[i]['keypoints_visible'][j]])
            bboxes = np.asarray([pose_results[i]['bboxes'][j]])
            data = {'bbox_scores':bbox_scores,
                    'keypoint_scores':keypoint_scores,
                    'keypoints':keypoints,
                    'keypoints_visible':keypoints_visible,
                    'bboxes':bboxes}
            if i == 0:
                pose_result_list.append([data])
            else:
                pose_result_list[j].append(data)
    return pose_result_list            



def main():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = frame_extract(args.video, args.short_side,
                                        tmp_dir.name)

    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                         frame_paths, args.det_score_thr,
                                         args.det_cat_id, args.device)

    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths, det_results,
                                                     args.device)
    
    # print(f"pose_data_samples[0].keys() : {pose_data_samples[0].keys()}")
    # print(f"len(pose_data_samples) : {len(pose_data_samples)}")

    torch.cuda.empty_cache()

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    model = init_recognizer(config, args.checkpoint, args.device)
    
    result = inference_skeleton(model, pose_results, (h, w))
    
    max_pred_index = result.pred_score.argmax().item()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    action_label = label_map[max_pred_index]

    visualize(args, frames, pose_data_samples, action_label)

    tmp_dir.cleanup()


# new Visualize()
def visualize_1(args, frames, data_samples, action_labels):
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(data_samples[0][0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton for each frame')
    for i, [d, f] in enumerate(track_iter_progress(list(zip(data_samples, frames)))):
        f = mmcv.imconvert(f, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
        vis_frame = visualizer.get_image()
        cv2.putText(vis_frame, action_labels[0], (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        vis_frames.append(vis_frame)

    vid = mpy.ImageSequenceClip(vis_frames, fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

def visualize_2(video, pose_results, ratio, action_labels, output_path):
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        for pr in range(len(pose_results)):
            pose = pose_results[pr][i]
            score = sum(pose_results[pr][i]['keypoint_scores'])
            if action_labels[pr][i] in ['chom nguoi ve phia truoc','kheu nguoi ban tren','cuoi xuong nhat tai lieu','xem tai lieu duoi bai thi','chom qua hai ben']:
                # if action_labels[pr][i+10] in ['chom nguoi ve phia truoc','kheu nguoi ban tren','cuoi xuong nhat tai lieu','xem tai lieu duoi bai thi','chom qua hai ben']:
                cv2.rectangle(frame, (int(pose['bboxes'][0][0]*ratio), int(pose['bboxes'][0][1]*ratio)), (int(pose['bboxes'][0][2]*ratio),int(pose['bboxes'][0][3]*ratio)), (0,0,255), 2)   
                # else:
                #     cv2.rectangle(frame, (int(pose['bboxes'][0][0]*ratio), int(pose['bboxes'][0][1]*ratio)), (int(pose['bboxes'][0][2]*ratio),int(pose['bboxes'][0][3]*ratio)), (0,255,0), 2) 
            else:
                cv2.rectangle(frame, (int(pose['bboxes'][0][0]*ratio), int(pose['bboxes'][0][1]*ratio)), (int(pose['bboxes'][0][2]*ratio),int(pose['bboxes'][0][3]*ratio)), (0,255,0), 2)
            
        out.write(frame)
            
    out.release()
    cv2.destroyAllWindows()

def timestamp_predict(model, timestamp, pose_results, h, w, label_map):
    """
    After each timestamp frame, the result of the action will be predicted according to that number of timestamp frame
    Args:
        model: action model
        timestamp: number of frame that I want to feed model
        pose_results: list of poses
    Returns:
        action_labels: [list] len(action_labels) = len(pose_result) - timestamp: list of the action_label in each frame
    """
    action_labels = []
    print('Predicting...')
    for ps in tqdm(range(len(pose_results))):
        for i in tqdm(range(len(pose_results[ps]))):
            pose_input = pose_results[ps][i:i+timestamp]
            result = inference_skeleton(model, pose_input, (h, w))
            max_pred_index = result.pred_score.argmax().item()
            labelmap = [x.strip() for x in open(label_map).readlines()]
            action_label = labelmap[max_pred_index]
            if i == 0:
                action_labels.append([action_label])
            else:
                action_labels[ps].append(action_label)
    
    return action_labels



# new Main()
def main_1():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    cap = cv2.VideoCapture(args.video)

    frame_paths, frames = frame_extract(args.video, args.short_side,
                                        tmp_dir.name)
    
    old_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_width = frames[0].shape[1]
    ratio = (old_width/new_width)

    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                         frame_paths, args.det_score_thr,
                                         args.det_cat_id, args.device)

    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths, det_results,
                                                     args.device)
    
    print(f"len(pose_results) : {len(pose_results)}")
    print(f"(pose_results[0]) : {(pose_results[0])}")
    print(f"pose_results[0].keys() : {pose_results[0].keys()}")

    torch.cuda.empty_cache()

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    # pose_result_list= post_processing(pose_results)
    model = init_recognizer(config, args.checkpoint, args.device)
    action_labels = []
    # for i in range(len(pose_results)):
    #     result = inference_skeleton(model, pose_results[i], (h, w))
    #     max_pred_index = result.pred_score.argmax().item()
    #     label_map = [x.strip() for x in open(args.label_map).readlines()]
    #     action_label = label_map[max_pred_index]
    #     action_labels.append(action_label)

    # visualize_2(args.video, pose_results, ratio, action_labels, args.out_filename)
    tmp_dir.cleanup()


def main_2():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    cap = cv2.VideoCapture(args.video)

    frame_paths, frames = frame_extract(args.video, args.short_side,
                                        tmp_dir.name)
    num_frames = len(frames)
    
    old_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_width = frames[0].shape[1]
    ratio = (old_width/new_width)

    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                         frame_paths, args.det_score_thr,
                                         args.det_cat_id, args.device)

    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference_2(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths, det_results,
                                                     args.device)
    
    dissected_pose_results = dissect_pose_results(pose_results)
    matched_results = match_pose(dissected_pose_results)
    

    torch.cuda.empty_cache()

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    model = init_recognizer(config, args.checkpoint, args.device)
    action_labels = timestamp_predict(model, 120, matched_results, h, w, args.label_map)
    visualize_2(args.video, matched_results, ratio, action_labels, args.out_filename)
    
    tmp_dir.cleanup()



if __name__ == '__main__':
    main_2()
