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

class App:
    def __init__(self):
        self.snake=Snake()
        self.detection_graph, sess = detector_utils.load_inference_graph()
        self.sess = tf.Session(graph=self.detection_graph)
        # dictionary to store parameters
        self.params = {
            "source": 0,
            "num_hands_detect": 2,
            "display_fps": True,
            "display_box": True,
            "score_thresh":0.2,
            "size":[900,1600]
        }
        self.video_capture = WebcamVideoStream(src=self.params["source"],size=self.params["size"]).start()
        self.params['im_width'], self.params['im_height'] = self.video_capture.size()

    def game_logic(self,hand_location,background):
        if hand_location is not None:
            self.snake.set_direction(hand_location)
        self.snake.move()
        for i in self.snake.get_position():
            cv2.circle(background,tuple(i),5,(255,255,255),-1)

    def wait_to_start(self,hand_location,head_location, head_size,background):

        if (hand_location[0]-head_location[0])>0 and abs(hand_location[1]-head_location[1])<head_size*1.5:
            self.game_start=True

    def start(self):
        #flag to check if game has started
        self.game_start=False
        num_frames = 0
        fps=0
        #use this flag to only process every second frame
        toggle=True
        start_time=time.time()
        head_location=None
        while True:
            #read video and background
            background=cv2.imread("images/background.jpg",1)
            frame = self.video_capture.read()

            background=cv2.resize(background,self.video_capture.size())
            num_frames+=1

            #toggle toggle
            toggle=1-toggle
            if toggle:
                continue

            frame = cv2.flip(frame, 1)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            if frame_small is None:
                break

            # actual detection
            #boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
            boxes, scores = detector_utils.detect_objects(frame_small, self.detection_graph, self.sess)
            # boxes=cv2.multiply(boxes,4)
            face_locations = face_recognition.face_locations(frame_small)
            # draw bounding boxes
            hand_location=detector_utils.draw_box_on_image(
                self.params['num_hands_detect'],
                self.params["score_thresh"], scores, boxes,
                self.params['im_width'], self.params['im_height'], background)

            for top, right, bottom, left in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw a box around the face
                cv2.rectangle(background, (left, top), (right, bottom), (0, 0, 255), 2)
                head_location=detector_utils.center_of(top,right,bottom,left)
                cv2.circle(background,head_location,5,(0,0,255),-1)

            if self.game_start:
                self.game_logic(hand_location,background)
            else:
                cv2.putText(background,'Raise hand to start',(10,360), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
                if head_location!=None and hand_location!=None:
                    self.wait_to_start(hand_location,head_location,right-left,background)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
            elapsed_time=(time.time()-start_time)
            if elapsed_time>1:
                fps = num_frames / elapsed_time

            if self.params["display_fps"]:
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), background)


            background[540:960, 720:1040]=frame_small
            cv2.imshow('Muilti - threaded Detection', background)
        self.close()

    def close(self):
        self.sess.close()
        self.video_capture.stop()
        cv2.destroyAllWindows()
app=App()
app.start()
