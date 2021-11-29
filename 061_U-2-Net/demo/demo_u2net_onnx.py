#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument(
        "--model",
        type=str,
        default='../01_float32/u2netp_320x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='320,320',
    )

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    elapsed_time = 0.0

    while True:
        start_time = time.time()

        # Capture read
        ret, image = cap.read()
        if not ret:
            break

        result_map = run_inference(
            onnx_session,
            input_size,
            image,
        )
        elapsed_time = time.time() - start_time

        # Inference elapsed time
        elapsed_time_text = "Elapsed time: "
        elapsed_time_text += str(round((elapsed_time * 1000), 1))
        elapsed_time_text += 'ms'
        cv.putText(image, elapsed_time_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 1, cv.LINE_AA)

        # Map Resize
        debug_image = cv.resize(result_map,
                                dsize=(image.shape[1], image.shape[0]))

        cv.imshow('U-2-Net Original', image)
        cv.imshow('U-2-Net Result', debug_image)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
