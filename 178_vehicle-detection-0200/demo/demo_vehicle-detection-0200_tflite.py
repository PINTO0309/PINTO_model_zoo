#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import math
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_256x256/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
    parser.add_argument(
        "--npy",
        type=str,
        default='saved_model_256x256/0.npy',
    )
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--nms_th", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run_inference(
    interpreter,
    input_size,
    image,
    prior_bbox,
    score_th,
    nms_th,
):
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    bbox_logits_list = interpreter.get_tensor(output_details[0]['index'])
    confidence_list = interpreter.get_tensor(output_details[1]['index'])

    # Post process
    bbox_logits_list = bbox_logits_list[0]
    confidence_list = confidence_list[0]
    bbox_list = []
    score_list = []
    # bbox decode
    for index in range(int(len(prior_bbox[0]) / 4)):
        score = confidence_list[index * 2 + 0]
        if score < score_th:
            continue

        prior_x0 = prior_bbox[0][index * 4 + 0]
        prior_y0 = prior_bbox[0][index * 4 + 1]
        prior_x1 = prior_bbox[0][index * 4 + 2]
        prior_y1 = prior_bbox[0][index * 4 + 3]
        prior_cx = (prior_x0 + prior_x1) / 2.0
        prior_cy = (prior_y0 + prior_y1) / 2.0
        prior_w = prior_x1 - prior_x0
        prior_h = prior_y1 - prior_y0

        box_cx = bbox_logits_list[index * 4 + 0]
        box_cy = bbox_logits_list[index * 4 + 1]
        box_w = bbox_logits_list[index * 4 + 2]
        box_h = bbox_logits_list[index * 4 + 3]

        prior_variance = [0.1, 0.1, 0.2, 0.2]
        cx = prior_variance[0] * box_cx * prior_w + prior_cx
        cy = prior_variance[1] * box_cy * prior_h + prior_cy
        w = math.exp((box_w * prior_variance[2])) * prior_w
        h = math.exp((box_h * prior_variance[3])) * prior_h

        image_height, image_width = image.shape[0], image.shape[1]
        bbox_list.append([
            int((cx - (w / 2.0)) * image_width),
            int((cy - (h / 2.0)) * image_height),
            int((cx - (w / 2.0)) * image_width) + int(w * image_width),
            int((cy - (h / 2.0)) * image_height) + int(h * image_height),
        ])
        score_list.append(float(score))
    # nms
    keep_index = cv.dnn.NMSBoxes(
        bbox_list,
        score_list,
        score_threshold=score_th,
        nms_threshold=nms_th,
        # top_k=200,
    )
    nms_bbox_list = []
    nms_score_list = []
    for index in keep_index:
        nms_bbox_list.append(bbox_list[index[0]])
        nms_score_list.append(score_list[index[0]])

    return nms_bbox_list, nms_score_list


def main():
    args = get_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    npy_path = args.npy
    score_th = args.score_th
    nms_th = args.nms_th

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 0.npy Load
    prior_bbox = np.squeeze(np.load(npy_path))

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        bboxes, scores = run_inference(
            interpreter,
            input_size,
            frame,
            prior_bbox,
            score_th,
            nms_th,
        )

        elapsed_time = time.time() - start_time

        # Draw bbox and score
        for bbox, score in zip(bboxes, scores):
            cv.putText(debug_image, '{:.3f}'.format(score), (bbox[0], bbox[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1,
                       cv.LINE_AA)
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 0, 0))

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('vehicle detection 0200', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
