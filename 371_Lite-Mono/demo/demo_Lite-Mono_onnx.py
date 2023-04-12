#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from typing import Tuple

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> np.ndarray:
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    depth_map = onnx_session.run(None, {input_name: input_image})

    # Post process
    depth_map = np.squeeze(depth_map[0])
    # d_min = np.min(depth_map)
    # d_max = np.max(depth_map)
    # depth_map = (depth_map - d_min) / (d_max - d_min)
    depth_map = depth_map * 255.0
    depth_map = np.asarray(depth_map, dtype="uint8")

    return depth_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-f', '--movie_file', type=str, default=None)
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='lite-mono_192x640.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap_device = args.device
    if args.movie_file is not None:
        cap_device = args.movie_file
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        depth_map = run_inference(
            onnx_session,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image, depth_image = draw_debug(
            debug_image,
            elapsed_time,
            depth_map,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('Input', debug_image)
        cv.imshow('Output', depth_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image: np.ndarray,
    elapsed_time: float,
    depth_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    # Apply ColorMap
    depth_image = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
    depth_image = cv.resize(depth_image, dsize=(image_width, image_height))

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image, depth_image


if __name__ == '__main__':
    main()