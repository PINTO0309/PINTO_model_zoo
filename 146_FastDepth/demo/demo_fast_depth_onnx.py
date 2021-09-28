#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process
    depth_map = result[0]
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    depth_map = (depth_map - d_min) / (d_max - d_min)
    depth_map = depth_map * 255.0
    depth_map = np.asarray(depth_map, dtype="uint8")

    depth_map = depth_map.reshape(input_size[0], input_size[1])

    return depth_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='fast_depth_128x160.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,160',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

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
        depth_map = run_inference(
            onnx_session,
            input_size,
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
        cv.imshow('FastDepth RGB Demo', debug_image)
        cv.imshow('FastDepth Depth Demo', depth_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, depth_map):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    # Apply ColorMap
    depth_image = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
    depth_image = cv.resize(depth_image, dsize=(image_width, image_height))

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image, depth_image


if __name__ == '__main__':
    main()