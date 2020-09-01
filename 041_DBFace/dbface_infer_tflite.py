import os
import sys
import math
import time

import numpy as np
from numpy.lib.stride_tricks import as_strided

import cv2

import argparse

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

def _exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [_exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([_exp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def IOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def NMS(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj[1], reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and IOU(obj[0], objs[j][0]) > iou:
                flags[j] = 1
    return keep


def max_pooling(x, kernel_size, stride=1, padding=1):
    x = np.pad(x, padding, mode='constant')
    output_shape = ((x.shape[0] - kernel_size)//stride + 1,
                    (x.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    x_w = as_strided(x, shape=output_shape + kernel_size, strides=(stride*x.strides[0], stride*x.strides[1]) + x.strides)
    x_w = x_w.reshape(-1, *kernel_size)

    return x_w.max(axis=(1, 2)).reshape(output_shape)


def detect(hm, box, landmark, threshold=0.4, nms_iou=0.5):
    hm_pool = max_pooling(hm[0,0,:,:], 3, 1, 1)        # 1,1,64,64
    interest_points = ((hm==hm_pool) * hm)             # screen out low-conf pixels
    flat            = interest_points.ravel()          # flatten
    indices         = np.argsort(flat)[::-1]           # index sort
    scores          = np.array([ flat[idx] for idx in indices ])

    hm_height, hm_width = hm.shape[1:3]
    ys = indices // hm_width
    xs = indices %  hm_width
    box      = box.reshape(box.shape[1:])           # 64,64,4
    landmark = landmark.reshape(landmark.shape[1:]) # 64,64,10

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold: 
            break
        x, y, r, b = box[cy, cx, :]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[cy, cx, :]
        x5y5 = (_exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append([xyrb, score, box_landmark])
    return NMS(objs, iou=nms_iou)


def drawBBox(image, bbox, scale_w, scale_h, color=(0,255,0), thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255)):

    text = f"{bbox[1]:.2f}"
    xyrb = bbox[0]
    x, y, r, b = int(xyrb[0] * scale_w), int(xyrb[1] * scale_h), int(xyrb[2] * scale_w), int(xyrb[3] * scale_h)
    w = r - x + 1
    h = b - y + 1

    cv2.rectangle(image, (x, y, r-x+1, b-y+1), color, thickness, 16)

    border = int(thickness / 2)
    pos = (x + 3, y - 5)
    cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

    landmark = bbox[2]
    if len(landmark)>0:
        for i in range(len(landmark)):
            x, y = landmark[i][:2]
            cv2.circle(image, (int(x * scale_w), int(y * scale_h)), 3, landmarkcolor, -1, 16)


def main(args):
    interpreter = Interpreter(model_path=args.model, num_threads=4)
    
    interpreter.allocate_tensors()
    inblobs = interpreter.get_input_details()
    outblobs = interpreter.get_output_details()

    # print('input:', inblobs[0]['shape'])
    # print('output0:', outblobs[0]['shape'])
    # print('output1:', outblobs[1]['shape'])
    # print('output2:', outblobs[2]['shape'])

    # input : [1, 256, 256, 3]
    # output: [1, 64, 64, 10], [1, 64, 64, 4], [1, 64, 64, 1]

    lm_idx  = 0
    box_idx = 1
    hm_idx  = 2

    fps = ""
    detectfps = ""
    framecount = 0
    detectframecount = 0
    time1 = 0

    mean = np.array([0.408, 0.447, 0.47], dtype="float32")
    std = np.array([0.289, 0.274, 0.278], dtype="float32")

    if args.input == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        start_time = time.perf_counter()
        if args.input == 'cam':
            ret, image = cap.read()
        else:
            image = cv2.imread(args.input)

        scale_w = image.shape[1] / inblobs[0]['shape'][2]
        scale_h = image.shape[0] / inblobs[0]['shape'][1]

        img = cv2.resize(image, (inblobs[0]['shape'][2], inblobs[0]['shape'][1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = ((img / 255.0 - mean) / std).astype(np.float32)
        img = img[np.newaxis,:,:,:]

        interpreter.set_tensor(inblobs[0]['index'], img)
        interpreter.invoke()

        lm = interpreter.get_tensor(outblobs[lm_idx]['index'])[0][np.newaxis,:,:,:]                          # 1,h,w,10
        box = interpreter.get_tensor(outblobs[box_idx]['index'])[0][np.newaxis,:,:,:]                        # 1,h,w,4
        hm = interpreter.get_tensor(outblobs[hm_idx]['index'])[0][np.newaxis,:,:,:].transpose((0,3,1,2))     # 1,1,h,w
        
        objs = detect(hm=hm, box=box, landmark=lm, threshold=0.4, nms_iou=0.5)

        for obj in objs:
            drawBBox(image, obj, scale_w, scale_h)

        cv2.putText(image, fps, (image.shape[1] - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('output', image)
        if args.input == 'cam':
            if cv2.waitKey(1) == 27:  # ESC key
                return
        else:
            cv2.waitKey(0)
            cv2.imwrite('output.jpg', image)
            print('"output.jpg" is generated')
            return

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps = "(Playback) {:.1f} FPS".format(time1 / 10)
            framecount = 0
            time1 = 0
        end_time = time.perf_counter()
        elapsedTime = end_time - start_time
        time1 += 1 / elapsedTime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='cam', help='input image file name (\'cam\' for webCam input)')
    parser.add_argument('-m', '--model', type=str, default='dbface_keras_256x256_integer_quant_nhwc.tflite', help='DBFace model file name (*.tflite)')
    args = parser.parse_args()

    main(args)
