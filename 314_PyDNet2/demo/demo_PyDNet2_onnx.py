#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image):
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, Normarize, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process
    depth_map = np.array(result)
    depth_map = np.squeeze(depth_map)
    depth_map = depth_map[:, :, 0]

    color_scaling = 1 / 64.0
    depth_map *= color_scaling
    depth_map = depth_map * 255.0
    depth_map = np.asarray(depth_map, dtype="uint8")

    return depth_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='pydnet2_192x512/model_float32.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
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
        cv.imshow('PyDNet2 RGB Demo', debug_image)
        cv.imshow('PyDNet2 Depth Demo', depth_image)

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
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image, depth_image


if __name__ == '__main__':
    main()
