#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import math
import argparse

import cv2 as cv
import numpy as np

from scrfd.scrfd_onnx import SCRFD


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_scrfd_500m_bnkps_480x640/scrfd_500m_bnkps_480x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )

    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--nms_th", type=float, default=0.4)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    score_th = args.score_th
    nms_th = args.nms_th

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    detector = SCRFD(model_file=model_path, nms_thresh=nms_th)
    detector.prepare(-1)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        bboxes, keypoints = detector.detect(
            frame,
            score_th,
            input_size=(input_size[1], input_size[0]),
        )

        elapsed_time = time.time() - start_time

        # Draw bbox and keypoint
        for index, bbox in enumerate(bboxes):
            x1, y1, x2, y2, _ = bbox.astype(np.int)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if keypoints is not None:
                for keypoint in keypoints[index]:
                    keypoint = keypoint.astype(np.int)
                    cv.circle(debug_image, tuple(keypoint), 5, (0, 0, 255), 2)

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('SCRFD', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
