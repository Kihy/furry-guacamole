from handtracking.utils import detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
import datetime
import argparse
from Webcam import WebcamVideoStream
import face_recognition
from snake import Snake
from snake_gui import *

"""Deprecated file"""

snake=Snake()
# dictionary to store parameters
detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
params = {
    "source": 0,
    "num_hands_detect": 2,
    "display_fps": True,
    "display_box": True,
    "score_thresh":0.2,
    "size":[900,1600]
}
video_capture = WebcamVideoStream(src=params["source"],size=params["size"]).start()
params['im_width'], params['im_height'] = video_capture.size()


start_time = datetime.datetime.now()
num_frames = 0
fps=0
#use this flag to only process every second frame
toggle=True
start_time=time.time()

while True:
    background=cv2.imread("images/background.jpg",1)
    frame = video_capture.read()

    background=cv2.resize(background,video_capture.size())
    num_frames+=1
    toggle=1-toggle
    if toggle:
        continue
    frame = cv2.flip(frame, 1)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_small=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    if (frame_small is not None):
        # actual detection
        #boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
        boxes, scores = detector_utils.detect_objects(frame_small, detection_graph, sess)
        # boxes=cv2.multiply(boxes,4)
        face_locations = face_recognition.face_locations(frame_small)
        # draw bounding boxes
        hand_location=detector_utils.draw_box_on_image(
            params['num_hands_detect'],
            params["score_thresh"], scores, boxes,
            params['im_width'], params['im_height'], background)

        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(background, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.circle(background,detector_utils.center_of(top,right,bottom,left),5,(0,0,255),-1)
        if hand_location is not None:
            snake.set_direction(hand_location)
        snake.move()
        for i in snake.get_position():
            cv2.circle(background,tuple(i),5,(255,255,255),-1)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
    elapsed_time=(time.time()-start_time)
    if elapsed_time>1:
        fps = num_frames / elapsed_time

    if params["display_fps"]:
        detector_utils.draw_fps_on_image(
            "FPS : " + str(int(fps)), background)


    background[540:960, 720:1040]=frame_small
    cv2.imshow('Muilti - threaded Detection', background)

sess.close()
video_capture.stop()
cv2.destroyAllWindows()
