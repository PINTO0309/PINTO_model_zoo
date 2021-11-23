#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import tensorflow as tf

from mlsd.utils_tflite import pred_lines


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_320x320_tiny/model_float16_quant.tflite')
    parser.add_argument("--input_shape", type=str, default='320,320')

    # M-LSD Parameters
    parser.add_argument("--score_thr", type=float, default=0.1)
    parser.add_argument("--dist_thr", type=float, default=20.0)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_device = args.device
    filepath = args.file

    model = args.model
    input_shape = [int(i) for i in args.input_shape.split(',')]
    score_thr = args.score_thr
    dist_thr = args.dist_thr

    # Initialize video capture
    cap = None
    if filepath is None:
        cap = cv.VideoCapture(cap_device)
    else:
        cap = cv.VideoCapture(filepath)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        lines = pred_lines(
            frame,
            interpreter,
            input_details,
            output_details,
            input_shape=input_shape,
            score_thr=score_thr,
            dist_thr=dist_thr,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            lines,
        )

        cv.imshow('M-LSD(Lines) Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    lines,
):
    # lines
    for line in lines:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv.line(image, (x_start, y_start), (x_end, y_end), [255, 0, 0], 2)

    # Inference elapsed time
    cv.putText(image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
