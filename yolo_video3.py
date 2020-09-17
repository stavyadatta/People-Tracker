import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from yolov3.yolov3 import Create_Yolov3
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names#, detect_image, detect_video, detect_realtime
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import argparse
import imutils
    
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
ap.add_argument('-o', "--output", required=True)
args = vars(ap.parse_args())

video_path = args["input"]
output_path = args["output"]

Darknet_weights = YOLO_V3_WEIGHTS
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)

if YOLO_CUSTOM_WEIGHTS != False:
    yolo.load_weights(YOLO_CUSTOM_WEIGHTS)
else:
    load_yolo_weights(yolo, Darknet_weights)
    
max_cosine_distance = 0.5
nn_budget = None
input_size = YOLO_INPUT_SIZE
score_threshold = 0.5
iou_threshold = 0.45
track_obj = 'person'


model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

times, times_2 = [], []

vid = cv2.VideoCapture(video_path)
prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
totalFrames = int(vid.get(prop))
print(f"info - Total frames are {totalFrames}")

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, codec, fps, (width, height))

NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
key_list = list(NUM_CLASS.keys())
val_list = list(NUM_CLASS.values())

while True:
    _, frame = vid.read()
    
    try:
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    except:
        break
    image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    tStart = time.time()
    pred_bbox = yolo.predict(image_data)
    
    tEnd = time.time()
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    # for removing extra boxes and detections using NMM 
    bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method="nms")
    boxes, scores, names = [], [], []
    
    for bbox in bboxes:
        if NUM_CLASS[int(bbox[5])] == track_obj:
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
            scores.append(bbox[4])
            names.append(NUM_CLASS[int(bbox[5])])
            
    boxes = np.array(boxes)
    names = np.array(names)
    scores = np.array(scores)
    features = np.array(encoder(original_frame, boxes))
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
    
    tracker.predict()
    tracker.update(detections)
    
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        tracking_id = track.track_id
        index = key_list[val_list.index(class_name)]
        tracked_bboxes.append(bbox.tolist() + [tracking_id, index])
        
    image = draw_bbox(original_frame, tracked_bboxes, CLASSES=YOLO_COCO_CLASSES, tracking=True)
    
    
    t3 = time.time()
    times.append(tEnd - tStart)
    times_2.append(t3 - tStart)
    ms = sum(times)/len(times_2)*1000

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
    
    image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    print("Time:{:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
    if output_path != '':
        out.write(image)
    
    cv2.destroyAllWindows()

    