#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import onnxruntime

from mlsd.utils_onnx import pred_squares


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--model",
                        type=str,
                        default='saved_model_320x320_tiny/model_float32.onnx')
    parser.add_argument("--input_shape", type=str, default='320,320')

    # M-LSD Parameters
    parser.add_argument("--score", type=float, default=0.1)
    parser.add_argument("--outside_ratio", type=float, default=0.1)
    parser.add_argument("--inside_ratio", type=float, default=0.5)
    parser.add_argument("--w_overlap", type=float, default=0.0)
    parser.add_argument("--w_degree", type=float, default=1.14)
    parser.add_argument("--w_length", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=1.84)
    parser.add_argument("--w_center", type=float, default=1.46)

    args = parser.parse_args()

    return args


def get_params(args):
    params = {
        'score': args.score,
        'outside_ratio': args.outside_ratio,
        'inside_ratio': args.inside_ratio,
        'w_overlap': args.w_overlap,
        'w_degree': args.w_degree,
        'w_length': args.w_length,
        'w_area': args.w_area,
        'w_center': args.w_center,
    }
    return params


def main():
    args = get_args()
    cap_device = args.device
    filepath = args.file

    model = args.model
    input_shape = [int(i) for i in args.input_shape.split(',')]

    # Initialize video capture
    cap = None
    if filepath is None:
        cap = cv.VideoCapture(cap_device)
    else:
        cap = cv.VideoCapture(filepath)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model)

    params = get_params(args)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        segments, squares, score_array, inter_points = pred_squares(
            debug_image,
            onnx_session,
            input_shape,
            params,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            segments,
            squares,
            score_array,
            inter_points,
        )

        cv.imshow('M-LSD(Squares) Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    segments,
    squares,
    score_array,
    inter_points,
):
    # segments
    for line in segments:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv.line(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    # score, squares
    for score, square in zip(score_array, squares):
        # score
        cv.putText(image, '{:.2f}'.format(score), (square[0][0], square[0][1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv.LINE_AA)

        # square
        cv.polylines(image, [square.reshape([-1, 1, 2])], True, (255, 0, 0), 2)
        for pt in square:
            cv.circle(image, (int(pt[0]), int(pt[1])), 6, (255, 0, 0), -1)

    # inter_points
    for pt in inter_points:
        x, y = [int(val) for val in pt]
        cv.circle(image, (x, y), 4, (0, 255, 0), -1)

    # Inference elapsed time
    cv.putText(image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
