from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import os
try:
    from armv7l.openvino.inference_engine import IENetwork, IECore
except:
    from openvino.inference_engine import IENetwork, IECore

fpsstr = ""
framecount = 0
time1 = 0

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, default="openvino/256x256/FP16/frozen_east_text_detection.xml", help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str, help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=256,	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=256, help="resized image height (should be multiple of 32)")
ap.add_argument("-cw", "--camera_width", type=int, default=640, help='USB Camera resolution (width). (Default=640)')
ap.add_argument("-ch", "--camera_height", type=int, default=480, help='USB Camera resolution (height). (Default=480)')
ap.add_argument('--device', type=str, default='CPU', help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                           Sample will look for a suitable plugin for device specified (CPU by default)')
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
# layerNames = [
# 	"feature_fusion/Conv_7/Sigmoid",
# 	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
model_xml = args["east"]
model_bin = os.path.splitext(model_xml)[0] + ".bin"
ie = IECore()
net = ie.read_network(model_xml, model_bin)
input_info = net.input_info
input_blob = next(iter(input_info))
exec_net = ie.load_network(network=net, device_name=args["device"])

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
    frame = imutils.resize(frame, width=args["camera_width"])
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
    frame = frame.astype(np.float32)
    frame -= mean
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = np.transpose(frame, [0, 3, 1, 2])
    predictions = exec_net.infer(inputs={input_blob: frame})
    scores = predictions['feature_fusion/Conv_7/Sigmoid']
    geometry = predictions['feature_fusion/concat_3']

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the frame
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
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