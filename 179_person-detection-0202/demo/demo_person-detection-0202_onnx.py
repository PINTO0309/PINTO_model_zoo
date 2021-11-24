#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import math
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_512x512/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,512',
    )
    parser.add_argument(
        "--npy",
        type=str,
        default='saved_model_512x512/0.npy',
    )
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--nms_th", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run_inference(
    onnx_session,
    input_size,
    image,
    prior_bbox,
    score_th,
    nms_th,
):
    # Pre process:Resize, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name01 = onnx_session.get_outputs()[0].name
    output_name02 = onnx_session.get_outputs()[1].name
    bbox_logits_list, confidence_list = onnx_session.run(
        [output_name01, output_name02],
        {input_name: input_image},
    )

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
    onnx_session = onnxruntime.InferenceSession(model_path)

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
            onnx_session,
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
        cv.imshow('person detection 0202', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
