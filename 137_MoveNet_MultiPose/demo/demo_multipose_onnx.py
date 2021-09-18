#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import tensorflow as tf


def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # Pre process:Resize, BGR->RGB, Reshape, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.reshape(-1, input_size[0], input_size[1], 3)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Postprocess:Calc Keypoint, boounding box
    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        # keypoint
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        # bounding box
        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        # Add data for 6 people to the list
        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_192x256/model_float32.onnx',
    )
    parser.add_argument("--input_size", type=str, default='192,256')
    parser.add_argument("--keypoint_score", type=float, default=0.3)
    parser.add_argument("--bbox_score_th", type=float, default=0.3)

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size.split(',')
    input_size = [int(i) for i in input_size]
    keypoint_score_th = args.keypoint_score
    bbox_score_th = args.bbox_score_th

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        keypoints_list, scores_list, bbox_list = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints_list,
            scores_list,
            bbox_score_th,
            bbox_list,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MoveNet(multipose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints_list,
    scores_list,
    bbox_score_th,
    bbox_list,
):
    debug_image = copy.deepcopy(image)

    connect_list = [
        [0, 1, (255, 0, 0)],  # nose → left eye
        [0, 2, (0, 0, 255)],  # nose → right eye
        [1, 3, (255, 0, 0)],  # left eye → left ear
        [2, 4, (0, 0, 255)],  # right eye → right ear
        [0, 5, (255, 0, 0)],  # nose → left shoulder
        [0, 6, (0, 0, 255)],  # nose → right shoulder
        [5, 6, (0, 255, 0)],  # left shoulder → right shoulder
        [5, 7, (255, 0, 0)],  # left shoulder → right elbow
        [7, 9, (255, 0, 0)],  # right elbow → left wrist
        [6, 8, (0, 0, 255)],  # right shoulder → right elbow
        [8, 10, (0, 0, 255)],  # right elbow → right wrist
        [11, 12, (0, 255, 0)],  # left hip → right hip
        [5, 11, (255, 0, 0)],  # left shoulder → left hip
        [11, 13, (255, 0, 0)],  # left hip → left knee
        [13, 15, (255, 0, 0)],  # left knee → left ankle
        [6, 12, (0, 0, 255)],  # right shoulder → right hip
        [12, 14, (0, 0, 255)],  # right hip → right knee
        [14, 16, (0, 0, 255)],  # right knee → right ankle
    ]

    for keypoints, scores in zip(keypoints_list, scores_list):
        # Connect Line
        for (index01, index02, color) in connect_list:
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, color, 2)

        # Keypoint circle
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv.circle(debug_image, keypoint, 3, (0, 255, 0), -1)

    # bounding box
    for bbox in bbox_list:
        if bbox[4] > bbox_score_th:
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0), 2)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()