import os
import cv2
import face_recognition
import tensorflow as tf
import numpy as np
import scipy.io
from handtracking.utils import detector_utils
import time
import matplotlib.pyplot as plt

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def test_face(iou_threshold=0.5,size=1):
    labels = open("test/face_label_no_small.txt", "r")
    path = labels.readline().rstrip()
    total = 0
    count = 0
    maximum = 0
    total_time=0
    num_pic=0
    minimum = float("inf")
    true_positive=0
    false_positive=0
    while path:
        image = cv2.imread('test/images/' + path, 1)
        num_pic+=1
        num_face = labels.readline().rstrip()
        image = cv2.resize(image, (0, 0), fx=size, fy=size,interpolation=cv2.INTER_CUBIC)
        start_time=time.time()
        face_locations = face_recognition.face_locations(image)
        total_time+=(time.time()-start_time)

        for i in range(int(num_face)):
            max_iou=0
            rect = labels.readline().rstrip().split()
            rect = list(map(int, rect))
            # cv2.rectangle(image,(rect[0],rect[1]),
            # (rect[2],rect[3]),(255,255,255))
            for top, right, bottom, left in face_locations:
                top*=(1/size)
                right*=(1/size)
                bottom*=(1/size)
                left*=(1/size)
                # cv2.rectangle(image,(int(left),int(top)),
                # (int(right),int(bottom)),(0,255,255))
                iou = IOU(rect, [left, top, right, bottom])
                max_iou=max(max_iou,iou)
            if max_iou<iou_threshold:
                false_positive+=1
            else:
                true_positive+=1
            total += max_iou
            minimum = min(minimum, max_iou)
            maximum = max(maximum, max_iou)
        # cv2.imshow('asdf', image)
        # cv2.waitKey(0)
        count += int(num_face)
        path = labels.readline().rstrip()
    print("face precision@{}: {:.3f}".format(iou_threshold, true_positive/(true_positive+false_positive)))
    print("face recall@{}: {:.3f}".format(iou_threshold,true_positive/count))
    print("average_time:",total_time/num_pic)
    return true_positive/(true_positive+false_positive),total_time/num_pic


def test_hand(iou_threshold=0.5,size=1):
    detection_graph, sess = detector_utils.load_inference_graph(
        "/frozen_inference_graph.pb")
    sess = tf.Session(graph=detection_graph)
    total_time=0
    num_pic=0
    labels = open("test/hand_label.txt", "r")
    path = labels.readline().rstrip()
    total = 0
    count = 0
    true_positive=0
    false_positive=0
    maximum = 0
    minimum = float("inf")
    while path:
        image = cv2.imread('test/image_hand/' + path, 1)
        num_pic+=1
        num_face = labels.readline().rstrip()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (0, 0), fx=size, fy=size)
        start_time=time.time()
        boxes, scores = detector_utils.detect_objects(
            image, detection_graph, sess)
        total_time+=(time.time()-start_time)
        _, hand_boxes = detector_utils.find_hand_in_image(
            0, 0.25, scores, boxes, image, False)

        rect_list=[]
        for i in range(int(num_face)):
            rect = labels.readline().rstrip().split()
            rect = list(map(int, rect))
            rect_list.append(rect)
        for top, right, bottom, left in hand_boxes:
            iou = 0
            # cv2.rectangle(image,(int(left),int(top)),
            # (int(right),int(bottom)),(0,255,255))
            top*=(1/size)
            right*=(1/size)
            bottom*=(1/size)
            left*=(1/size)
            for rect in rect_list:
                # cv2.rectangle(image,(rect[0],rect[1]),
                # (rect[2],rect[3]),(255,255,255))
                iou = max(iou, IOU(rect, [left, top, right, bottom]))
            if iou<iou_threshold:
                false_positive+=1
            else:
                true_positive+=1
            total += iou
            minimum = min(minimum, iou)
            maximum = max(maximum, iou)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('asdf',image)
        # cv2.waitKey(0)

        count += int(num_face)
        path = labels.readline().rstrip()

    print("hand precision@{}: {:.3f}".format(iou_threshold, true_positive/(true_positive+false_positive)))
    print("hand recall@{}: {:.3}".format(iou_threshold,true_positive/count))
    print("average_time:",total_time/num_pic)
    return true_positive/(true_positive+false_positive), total_time/num_pic
def test_scale():
    data={'average_time':[],
    'picture_scale':[],
    'precision':[]}
    for scale in np.arange(0.2,1.1,0.1):
        precision,av_time=test_face(size=scale)
        data['average_time'].append(av_time)
        data['picture_scale'].append(scale)
        data['precision'].append(precision)
    plt.plot('picture_scale','average_time',data=data,label='average_time')
    plt.plot('picture_scale','precision',data=data,label='precision')
    plt.ylabel('precision/time(s)')
    plt.xlabel('picture scale')
    plt.title('Effect of picture size in hand detector with large pictures')
    plt.legend()
    plt.show()
test_scale()
