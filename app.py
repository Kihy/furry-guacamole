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
        """Initialize the program"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_color = (255, 255, 255)  # white
        self.snake_color = (0, 255, 0)
        self.user_pic_size = (96, 96)
        # self.mouse_icon=cv2.imread("images/mouse.png",1)
        self.user_pic = None
        self.init_time = None #measures how long the hand is over the head
        self.hand_area = None #coordinates of control rectangle
        self.game_time = None #time for playing the game
        self.score = 0
        self.food_list = []

        self.high_score = 0
        self.high_score_pic = None

        #initializes the cnn and load data
        self.detection_graph, sess = detector_utils.load_inference_graph("/frozen_inference_graph.pb")
        self.sess = tf.Session(graph=self.detection_graph)
        self.game_start = False
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
        self.snake = Snake(self.screen_size[0], self.screen_size[1]) #create a snake

    def reset(self):
        """resets parameters once the game is finished"""
        self.user_pic = None
        self.init_time = None
        self.hand_area = None
        self.game_time = None
        self.score = 0
        self.food_list = []
        self.snake = Snake(self.screen_size[0], self.screen_size[1])
        self.game_start = False

    def find_hand(self,hand_locations):
        """finds the first hand inside the control rectangle"""
        for hand in hand_locations:
            if self.hand_area[0]<hand[1]<self.hand_area[2] and self.hand_area[1]<hand[0]<self.hand_area[3]:
                return hand
        return None

    def game_logic(self, hand_locations, background, elapsed_time,hand_boxes):
        """The game engine"""
        #if there are less than 5 food spots generate new one
        if len(self.food_list) < 5:
            self.generate_food()

        #if hand is detected
        if len(hand_locations) != 0:
            #find a hand inside the control rectangle
            hand=self.find_hand(hand_locations)
            #if a hand is found
            if hand is not None:
                #project the hand position onto screen, draw it and set the
                #direction of the snake
                pointer_location = self.hand_projection(hand)
                cv2.circle(background, pointer_location, 5, (165, 165, 42), -1)
                self.snake.set_direction(pointer_location, elapsed_time)
            else:
                #else warn the user and draw the control rectangle and all possible hands
                self.center_text_X('hand outside control box',background, 200)
                cv2.rectangle(background, (self.hand_area[1], self.hand_area[0]), (
                    self.hand_area[3], self.hand_area[2]), self.font_color, 3, 1)
                for top,right,bottom,left in hand_boxes:
                    p1 = (int(left), int(top))
                    p2 = (int(right), int(bottom))
                    cv2.rectangle(background, p1, p2, (77, 255, 9), 3, 1)
        else:
            #warn the user no hand is detected
            self.center_text_X('No hand detected', background, 200)
        #move the snake
        self.snake.move()
        # draw snake
        for i in self.snake.get_position():
            cv2.circle(background, tuple(i), 5, self.snake_color, -1)
        # check food is eaten
        for i in self.food_list:
            if self.is_within(self.snake.head_pos(), i, 20):
                self.food_list.remove(i)
                self.score += 1
        # draw food
        for i in self.food_list:
            cv2.circle(background, i, 10, (239, 35, 66), -1)

    def is_within(self, pos1, pos2, delta):
        """checks if two point is within delta range"""
        return np.linalg.norm(pos1 - pos2) < delta

    def generate_food(self):
        """creates random food spot in visible area"""
        x = random.randint(self.user_pic_size[1], int(
            self.screen_size[0] * 0.75))
        y = random.randint(self.user_pic_size[0], int(
            self.screen_size[1] * 0.75))
        self.food_list.append((x, y))

    def wait_to_start(self, hand_locations, head_locations, frame, hand_boxes):
        """initialize stage"""
        #for each head location
        for top, right, bottom, left in head_locations:
            #for each hand location
            for i in range(len(hand_locations)):
                hand = hand_locations[i]
                #if hand is within 2*head's width and over the head
                if hand[1] < top and abs(hand[0] - (left + right) / 2) < (right - left) * 2:
                    # start the timer if not already started
                    if self.init_time is None:
                        self.init_time = time.time()
                    # if more than 3 seconds passed
                    if time.time() - self.init_time > 3:
                        #start the game, save user's face and set coordinates of
                        #the control rectangle
                        self.game_start = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.user_pic = cv2.resize(
                            frame[top:left, bottom:right], self.user_pic_size)
                        self.hand_height = int(
                            (hand_boxes[i][2] - hand_boxes[i][0]) * 0.75)
                        self.hand_width = int(
                            self.hand_height * self.screen_ratio)
                        self.hand_area = [hand[1] - self.hand_height, hand[0] - self.hand_width,
                                          hand[1] + self.hand_height, hand[0] + self.hand_width]
                else:
                    #else reset timer
                    self.init_time = None


    def hand_projection(self, hand_location):
        """projects hand_location in control rectangle to screen coordinates"""
        x = hand_location[0] - self.hand_area[1]
        y = hand_location[1] - self.hand_area[0]

        return int(x * self.screen_size[1] / (self.hand_height * 2)), int(y * self.screen_size[0] / (self.hand_width * 2))

    def detect_face(self, background, frame_small):
        """detects a face with HOG algorithm, sorted by area of face detected"""
        face_locations = face_recognition.face_locations(frame_small)
        head_locations = []
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if not self.game_start:
                # Draw a box around the face
                cv2.rectangle(background, (left, top),
                              (right, bottom), (0, 0, 255), 2)
            head_location = [top, right, bottom, left]
            head_locations.append(head_location)
        return sorted(head_locations,key=lambda x:(x[2]-x[0])*(x[1]-x[3]))

    def detect_hands(self, background, frame):
        """hand detector with SSD"""
        # actual detection
        #boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
        boxes, scores = detector_utils.detect_objects(
            frame, self.detection_graph, self.sess)
        # boxes=cv2.multiply(boxes,4)

        # find hands

        return detector_utils.find_hand_in_image(
            self.params['num_hands_detect'],
            self.params["score_thresh"], scores, boxes, background, not self.game_start)


    def center_text_X(self, text, background, y):
        """centeres texts on the screen"""
        text_size = cv2.getTextSize(text, self.font, 1, 2)[0]

        cv2.putText(background, text, ((background.shape[1] - text_size[0]) // 2, y),
                    self.font, 1, self.font_color, 2, cv2.LINE_AA)

    def start(self):
        """The main game loop"""
        # flag to check if game has started
        self.game_start = False
        frames = 0
        # use this flag to only process every second frame
        toggle = True
        # used for fps calculation
        start_time2=time.time()
        head_location = None
        #start the main loop
        while True:
            #used for elapsed_time calculation
            start_time = time.time()
            # read video and background
            background = cv2.imread("images/background.jpg", 1)
            frame = self.video_capture.read()
            frames += 1
            #resize the background image to video size
            background = cv2.resize(background, self.video_capture.size())

            # toggle toggle
            toggle = 1 - toggle
            if toggle:
                continue

            #convert color for CNN
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #shrink the image to a quarter
            frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            if frame_small is None:
                break

            hand_locations, hand_boxes = self.detect_hands(background, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
            elapsed_time = (time.time() - start_time)

            fps = frames / (time.time()-start_time2)

            #if game is started
            if self.game_start:
                #start game timer
                if self.game_time is None:
                    self.game_time = time.time()
                #display countdown
                countdown = max(0,  60 - (time.time() - self.game_time))
                self.center_text_X('Time Remaining: {:.0f}'.format(
                    countdown), background, 50)
                self.center_text_X('Score: {}'.format(
                    self.score), background, 100)

                #call game logic
                self.game_logic(hand_locations, background, elapsed_time,hand_boxes)
                #add current user's image onto the screen
                pic_size = self.user_pic.shape
                background[0:pic_size[0], 0:pic_size[1]] = self.user_pic

                #if game is finished, determine high score and reset
                if time.time() - self.game_time > 60:
                    if self.high_score < self.score:
                        self.high_score = self.score
                        self.high_score_pic = self.user_pic
                    self.reset()
            else:
                #find all faces
                face_locations = self.detect_face(background, frame_small)
                #start 3 second count down
                if self.init_time is None:
                    time_remain = 3
                else:
                    time_remain = max(0, 3 - (time.time() - self.init_time))
                #display instructions to start the game
                self.center_text_X(
                    'Raise hand to a comfortable position', background, 280)
                self.center_text_X('above your head for {:.0f} seconds to start'.format(
                    time_remain), background, 350)

                #check if any face or hand is detected
                if len(face_locations) != 0:
                    if len(hand_locations) != 0:
                        self.wait_to_start(
                            hand_locations, face_locations, frame, hand_boxes)
                    else:
                        self.center_text_X('No hand detected', background, 100)
                else:
                    self.center_text_X('No face detected', background, 100)

            # display fps
            if self.params["display_fps"]:
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), background)

            #display current high score
            if self.high_score_pic is not None:
                # TODO: place this at better place
                cv2.putText(background, 'High Score: {}'.format(self.high_score), (0, background.shape[0] - 30 - self.high_score_pic.shape[0]),
                            self.font, 1, self.font_color, 2, cv2.LINE_AA)
                background[background.shape[0] - self.high_score_pic.shape[0]:background.shape[0],
                           0:self.high_score_pic.shape[1]] = self.high_score_pic

            #add current webcam to the screen
            frame_small_size = frame_small.shape
            background_size = background.shape
            background[background_size[0] - frame_small_size[0]:background_size[0],
                       background_size[1] - frame_small_size[1]:background_size[1]] = frame_small
            cv2.imshow('Le Snake', background)

        #gracefully close everything
        self.close()

    def close(self):
        """gracefully close everything"""
        self.sess.close()
        self.video_capture.stop()
        cv2.destroyAllWindows()


app = App()
app.start()
