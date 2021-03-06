import cv2
from threading import Thread
# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/


class WebcamVideoStream:
    def __init__(self, src, size=None):
        # initialize the video camera stream and read the first frame
        # from the stream

        self.stream = cv2.VideoCapture(src)

        if size != None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def ratio(self):
        return float(self.stream.get(3))/float(self.stream.get(4))

    def size(self):
        # return size of the capture device
        return int(self.stream.get(3)), int(self.stream.get(4))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
