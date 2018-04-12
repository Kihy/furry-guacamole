import cv2
import operator
import numpy as np

def normalize(direction):
    norm = np.linalg.norm(direction)
    return direction/norm

class Snake:
    def __init__(self):
        self.length=5;
        self.position=[np.array([i,i]) for i in range(self.length)]
        self.speed=np.array([50,50])#speed per second
        self.direction=np.array([0,0])

    def set_direction(self,hand_location,elapsed_time):
        direction=np.array(hand_location)- self.head_pos()
        direction=self.speed*elapsed_time*normalize(direction)
        self.direction=np.rint(direction).astype(int)

    def move(self):
        next_point= self.position[0]+self.direction
        self.position.insert(0,next_point)
        if len(self.position)>self.length:
            self.position.pop()

    def get_position(self):
        return self.position

    def head_pos(self):
        return self.position[0]
