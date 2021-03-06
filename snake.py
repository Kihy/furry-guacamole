import cv2
import operator
import numpy as np

def normalize(direction):
    """Normalizes the direction vector"""
    norm = np.linalg.norm(direction)
    return direction/norm

class Snake:
    def __init__(self,x,y):
        """
        initialize the snake
        x: x coordinate of snake head
        y: y coordinate of snake head
        """
        self.x=x
        self.y=y
        self.length=5;
        self.position=[np.array([100+i,100+i]) for i in range(self.length)]
        self.speed=np.array([50,50])#speed per second
        self.direction=np.array([0,0])

    def set_direction(self,hand_location,elapsed_time):
        """
        sets the direction of the snake
        hand_location: location user's hand is pointing
        elapsed_time: time used to calculate how far the snake should go
        """
        direction=np.array(hand_location)- self.head_pos()
        direction=self.speed*elapsed_time*normalize(direction)
        self.direction=np.rint(direction).astype(int)

    def move(self):
        """
        moves the snake according to direction
        """
        next_point= self.position[0]+self.direction

        next_point[0]%=self.x
        next_point[1]%=self.y
        self.position.insert(0,next_point)
        if len(self.position)>self.length:
            self.position.pop()

    def get_position(self):
        return self.position

    def head_pos(self):
        return self.position[0]
