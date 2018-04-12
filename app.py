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
import random
import numpy as np

class App:
    def __init__(self):
        self.user_pic = None
        self.init_time = None
        self.hand_area = None
        self.game_time=None
        self.score=0
        self.food_list=[]
        self.snake = Snake()
        self.detection_graph, sess = detector_utils.load_inference_graph()
        self.sess = tf.Session(graph=self.detection_graph)
        # dictionary to store parameters
        self.params = {
            "source": 0,
            "num_hands_detect": 2,
            "display_fps": True,
            "display_box": True,
            "score_thresh": 0.2,
            "size": [900, 1600]
        }
        self.video_capture = WebcamVideoStream(
            src=self.params["source"], size=self.params["size"]).start()
        self.screen_ratio = self.video_capture.ratio()
        self.screen_size = self.video_capture.size()

    def game_logic(self, hand_locations, background, elapsed_time):
        if int(time.time()-self.game_time)%3==0:
            self.generate_food()
        if len(hand_locations) != 0:
            pointer_location = self.hand_projection(hand_locations[0])
            cv2.circle(background, pointer_location, 5, (0, 0, 0), -1)
            self.snake.set_direction(pointer_location, elapsed_time)
        self.snake.move()
        # draw snake
        for i in self.snake.get_position():
            cv2.circle(background, tuple(i), 5, (255, 255, 255), -1)
        #check food is eaten
        for i in range(len(self.food_list)):
            #todo:lakdfjalkd
            if self.is_within(self.snake.head_pos(),self.food_list[i],10):
                self.food_list.pop(i)
                self.score+=1
        #draw food
        for i in self.food_list:
            cv2.circle(background, i, 10, (239,35,66), -1)


    def is_within(self,pos1,pos2,delta):
        return np.linalg.norm(pos1-pos2)<delta

    def generate_food(self):
        x=random.randint(self.user_pic.shape[1],int(self.screen_size[1]*0.75))
        y=random.randint(self.user_pic.shape[0],int(self.screen_size[0]*0.75))
        self.food_list.append((x,y))

    def wait_to_start(self, hand_locations, head_locations, frame, hand_boxes):
        for top, right, bottom, left in head_locations:
            for i in range(len(hand_locations)):
                hand = hand_locations[i]
                if hand[1] < top and abs(hand[0] - (left + right) / 2) < (right - left) * 2:
                    if self.init_time is None:
                        self.init_time = time.time()
                    if time.time() - self.init_time > 3:
                        self.game_start = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.user_pic = frame[top:left, bottom:right]
                        self.hand_height = int(
                            (hand_boxes[i][2] - hand_boxes[i][0])*0.75)
                        self.hand_width = int(
                            self.hand_height * self.screen_ratio)
                        self.hand_area = [hand[1] - self.hand_height, hand[0] - self.hand_width,
                                          hand[1] + self.hand_height, hand[0] + self.hand_width]
                else:
                    self.init_time = None

    def hand_projection(self, hand_location):
        x = hand_location[0] - self.hand_area[1]
        y = hand_location[1] - self.hand_area[0]

        return int(x * self.screen_size[1] / (self.hand_height * 2)), int(y * self.screen_size[0] / (self.hand_width * 2))

    def detect_face(self, background, frame_small):
        face_locations = face_recognition.face_locations(frame_small)
        head_locations = []
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(background, (left, top),
                          (right, bottom), (0, 0, 255), 2)
            head_location = [top, right, bottom, left]
            head_locations.append(head_location)
        return head_locations

    def detect_hands(self, background, frame):
        # actual detection
        #boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
        boxes, scores = detector_utils.detect_objects(
            frame, self.detection_graph, self.sess)
        # boxes=cv2.multiply(boxes,4)

        # draw bounding boxes
        return detector_utils.draw_box_on_image(
            self.params['num_hands_detect'],
            self.params["score_thresh"], scores, boxes, background)

    def start(self):
        # flag to check if game has started
        self.game_start = False
        fps = 0
        # use this flag to only process every second frame
        toggle = True

        head_location = None
        while True:
            start_time = time.time()
            # read video and background
            background = cv2.imread("images/background.jpg", 1)
            frame = self.video_capture.read()

            background = cv2.resize(background, self.video_capture.size())

            # toggle toggle
            toggle = 1 - toggle
            if toggle:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            if frame_small is None:
                break

            hand_locations, hand_boxes = self.detect_hands(background, frame)
            face_locations = self.detect_face(background, frame_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
            elapsed_time = (time.time() - start_time)

            fps = 1 / elapsed_time

            if self.game_start:
                if self.game_time is None:
                    self.game_time=time.time()
                countdown = max(0,  60- (time.time() - self.game_time))
                cv2.putText(background, 'Time Remaining: {:.0f}'.format(countdown), (20, self.user_pic.shape[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(background, 'Score: {}'.format(self.score), (20, self.user_pic.shape[1]+100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                self.game_logic(hand_locations, background, elapsed_time)
                pic_size = self.user_pic.shape
                background[0:pic_size[0], 0:pic_size[1]] = self.user_pic
                cv2.rectangle(background, (self.hand_area[1], self.hand_area[0]), (
                    self.hand_area[3], self.hand_area[2]), (255, 255, 255), 3, 1)

            else:
                cv2.putText(background, 'Raise hand to a comfortable position', (10, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                if self.init_time is None:
                    time_remain = 3
                else:
                    time_remain = max(0, 3 - (time.time() - self.init_time))
                cv2.putText(background, 'above your head for {:.0f} seconds to start'.format(time_remain), (10, 480),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                if len(face_locations) != 0 and len(hand_locations) != 0:
                    self.wait_to_start(
                        hand_locations, face_locations, frame, hand_boxes)

            if self.params["display_fps"]:
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), background)

            frame_small_size = frame_small.shape
            background_size = background.shape
            background[background_size[0] - frame_small_size[0]:background_size[0],
                       background_size[1] - frame_small_size[1]:background_size[1]] = frame_small
            cv2.imshow('Le Snake', background)

        self.close()

    def close(self):
        self.sess.close()
        self.video_capture.stop()
        cv2.destroyAllWindows()


app = App()
app.start()
