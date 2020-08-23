from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.basic import edgetpu_utils
import math

fpsstr = ""
framecount = 0
time1 = 0

def rotated_Rectangle(img, rotatedRect, color, thickness=1, lineType=cv2.LINE_8, shift=0):
    (x, y), (width, height), angle = rotatedRect
 
    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))
 
    t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                    [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                    [0,             0,              1]])
 
    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))
 
    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))
 
    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))
 
    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))
 
    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

    return points
 

def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), angles[pick]

def decode_predictions(scores, geometry1, geometry2):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    angles = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry1[0, 0, y]
        xData1 = geometry1[0, 1, y]
        xData2 = geometry1[0, 2, y]
        xData3 = geometry1[0, 3, y]
        anglesData = geometry2[0, 0, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of the bounding box
            h = (xData0[x] + xData2[x])
            w = (xData1[x] + xData3[x])
  
            # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            angles.append(angle)

	# return a tuple of the bounding boxes and associated confidences
    return (rects, confidences, angles)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, default="east_text_detection_320x320_full_integer_quant_edgetpu.tflite", help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str, help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
ap.add_argument("-cw", "--camera_width", type=int, default=640, help='USB Camera resolution (width). (Default=640)')
ap.add_argument("-ch", "--camera_height", type=int, default=480, help='USB Camera resolution (height). (Default=480)')
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
engine = BasicEngine(model_path=args["east"], device_path=devices[0])
offset = 0
output_offsets = [0]
for size in engine.get_all_output_tensors_sizes():
    offset += int(size)
    output_offsets.append(offset)

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    t1 = time.perf_counter()

    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=640)
    orig = frame.copy()

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio
    frame = cv2.resize(frame, (newW, newH))

    # construct a blob from the frame and then perform a forward pass
    # of the model to obtain the two output layer sets
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.uint8)
    inference_time, output = engine.run_inference(frame.flatten())
    outputs = [output[i:j] for i, j in zip(output_offsets, output_offsets[1:])]
    scores = outputs[0].reshape(1, int(args["height"]/4), int(args["width"]/4), 1)
    geometry1 = outputs[1].reshape(1, int(args["height"]/4), int(args["width"]/4), 4)
    geometry2 = outputs[2].reshape(1, int(args["height"]/4), int(args["width"]/4), 1)
    scores = np.transpose(scores, [0, 3, 1, 2])
    geometry1 = np.transpose(geometry1, [0, 3, 1, 2])
    geometry2 = np.transpose(geometry2, [0, 3, 1, 2])

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences, angles) = decode_predictions(scores, geometry1, geometry2)
    boxes, angles = non_max_suppression(np.array(rects), probs=confidences, angles=np.array(angles))

    # loop over the bounding boxes
    for ((startX, startY, endX, endY), angle) in zip(boxes, angles):
        # scale the bounding box coordinates based on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the frame
        width   = abs(endX - startX)
        height  = abs(endY - startY)
        centerX = int(startX + width / 2)
        centerY = int(startY + height / 2)

        rotatedRect = ((centerX, centerY), ((endX - startX), (endY - startY)), -angle)
        points = rotated_Rectangle(orig, rotatedRect, color=(0, 255, 0), thickness=2)
        cv2.polylines(orig, [points], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_8, shift=0)
        cv2.putText(orig, fpsstr, (args["camera_width"]-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Text Detection", orig)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

    # FPS calculation
    framecount += 1
    if framecount >= 10:
        fpsstr = "(Playback) {:.1f} FPS".format(time1/10)
        framecount = 0
        time1 = 0
    t2 = time.perf_counter()
    elapsedTime = t2-t1
    time1 += 1/elapsedTime

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()