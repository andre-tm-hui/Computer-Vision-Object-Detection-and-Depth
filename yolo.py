################################################################################

# Performs yolo on a single image-disparity map pair, to return a notated image
# portraying the depth of each detected object.

# Author : Andre Hui

# Implements the You Only Look Once (YOLO) object detection architecture decribed in full in:
# Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
# https://pjreddie.com/media/files/papers/YOLOv3.pdf

# This code: significant portions based in part on the tutorial and example available at:
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
# under LICENSE: https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/LICENSE

# To use first download the following files:

# https://pjreddie.com/media/files/yolov3.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

################################################################################

import cv2
import sys
import math
import numpy as np
import time
import torch
from models import *

################################################################################
# Camera calibration values

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

#####################################################################
# Gamma correction function

def adj_gamma(img, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img, table)

#####################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

################################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, depth, toggleConfidence):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, depth) + 'm'
    # add confidence value if enabled
    if toggleConfidence:
        label += ':%s' % (round(confidence,2))

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.375, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1*labelSize[1])),
        (left + round(1*labelSize[0]), top + baseLine), colour, cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.375, (255,255,255), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores
    try:
        for detection in results:
            confidence = detection[4]
            if confidence > threshold_confidence:
                width = int(detection[2]-detection[0])
                height = int(detection[3]-detection[1])
                left = int(detection[0])
                top = int(detection[1])
                classIds.append(int(detection[6]))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    except:
        for result in results:
            for detection in result:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > threshold_confidence:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds, confidences, boxes)


################################################################################
# rescale detected object boxes to original size

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # calculating x and y extension ratios
    ext_x = orig_w / current_dim
    ext_y = orig_h / current_dim
    # Rescale bounding boxes to dimension of original image
    for i in range(len(boxes)):
        boxes[i][0] *= ext_x
        boxes[i][1] *= ext_y
        boxes[i][2] *= ext_x
        boxes[i][3] *= ext_y
    return boxes

#####################################################################
# init YOLO CNN object detection model using pytorch
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

with open("coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

if torch.cuda.is_available():
    device = torch.device("cuda")

    # set up model
    model = Darknet("yolov3.cfg", 416).to(device)
    # load darknet weights
    model.load_darknet_weights("yolov3.weights")
    # set to evaluation mode
    model.eval()

    Tensor = torch.cuda.FloatTensor
else:
    # load configuration and weight files for the model and load the network using them
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    output_layer_names = getOutputsNames(net)

     # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

    # change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


#####################################################################
# find interval of highest frequency from a box of disparities
# first, we get focal length and baseline distance, and then flatten and remove the '0's from the disparity map
# leaving us with a 1D array. We then check that the array is non-empty, returning 0 if it is

def min_distance(disparity):
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    disparity = disparity.flatten()
    disparity = disparity[disparity != 0]

    if len(disparity) > 0:
        # find the modal value from the array
        out = np.bincount(disparity).argmax()
        return (f * B) / out
    else:
        return 0

def hist_distance(disparity, binSize, useDistances):
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    disparity = disparity.flatten()
    disparity = disparity[disparity != 0]

    if len(disparity) == 0:
        return 0

    # set histogram interval size, and create a list of interval start points that cover the range of disparity values
    bins = np.arange(min(disparity)//binSize*binSize,max(disparity) + binSize,binSize)

    # set up a dictionary for better organisation
    hist = {}
    for b in bins:
        hist[b] = []

    # keep track of the largest bin
    maxSize = 0
    biggest = []

    # populate the histogram with all the values
    for d in disparity:
        hist[d//binSize*binSize] += [d]

    # find the largest bin
    for h in hist:
        if len(hist[h]) > maxSize:
            maxSize = len(hist[h])
            biggest = hist[h]

    if useDistances:
        return np.mean(biggest)
    else:
        return (f * B) / np.mean(biggest)


#####################################################################
# function to draw boxes on a frame, given the disparity map

def drawBoxes(frame, disparity, useHist, removeOverlap, toggleConfidence, useDistances):
    # pre-process the image with an unsharp mask
    frame_copy = np.copy(frame)
    temp = cv2.GaussianBlur(frame_copy,(11,11),10)
    frame_copy = cv2.addWeighted(frame_copy,2,temp,-0.5,0)
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = cv2.dnn.blobFromImage(frame_copy, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    
    if torch.cuda.is_available():
        # convert openCV blob to a tensor object for pytorch
        img_ = torch.from_numpy(tensor).float()
        img_ = Variable(img_.type(Tensor))

        # get the detections and do non-maximal suppression
        with torch.no_grad():
            results = model(img_)
            results = non_max_suppression(results, confThreshold, nmsThreshold)
        # check for detections
        try:
            results = rescale_boxes(results[0].numpy(), inpWidth, frame.shape[:2])
            # separate results into classIDs, confidences, boxes
            classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)
        except:
            boxes = []
    else:
        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)
    
    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        if removeOverlap:
            # set all other boxes in the disparity matrix to contain '0' values, effectively removing any possible overlaps
            tempDisparity = np.copy(disparity)
            for obj in range(0, len(boxes)):
                if obj != detected_object:
                    b = boxes[obj]
                    l = b[0]
                    t = b[1]
                    w = b[2]
                    h = b[3]
                    tempDisparity[max(0,t):t+h, max(0,l):l+w] = 0

            disparityBox = tempDisparity[max(0,top):top+height, max(0,left):left+width]
        else:
            disparityBox = disparity[max(0,top):top+height, max(0,left):left+width]

        # calculate the distance to an object
        if useDistances:
            """temp = disparityBox.flatten()
            temp = temp[temp != 0]
            if len(temp) > 0:
                dist = np.mean(temp)
            else:
                dist = 0"""
            dist = hist_distance(disparityBox, 3, True)
        else:
            if useHist:
                dist = hist_distance(disparityBox, 5, False)
            else:
                dist = min_distance(disparityBox)

            # check if distance is 0, and retry with original disparity box if so
            if dist == 0:
                if useHist:
                    dist = hist_distance(disparity[max(0,top):top+height, max(0,left):left+width], 5, True)
                else:
                    dist = min_distance(disparity[max(0,top):top+height, max(0,left):left+width])

        # draw resulting detections on image, if the distance is greater than 0
        if dist > 0:
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50), dist, toggleConfidence)
    
    return frame

################################################################################
