#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, uint8 cast
    result_r = np.array(result[0])
    result_t = np.array(result[1])

    output_image_t = result_t[0]
    output_image_t = np.clip(output_image_t * 255.0, 0, 255).astype(np.uint8)

    output_image_r = result_r[0]
    output_image_r = np.clip(output_image_r * 255.0, 0, 255).astype(np.uint8)

    return output_image_t, output_image_r


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image_t, output_image_r = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        output_image_t = cv.resize(output_image_t,
                                   dsize=(frame_width, frame_height))
        output_image_r = cv.resize(output_image_r,
                                   dsize=(frame_width, frame_height))
        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('perceptual-reflection-removal Input', debug_image)
        cv.imshow('perceptual-reflection-removal Transmission', output_image_t)
        cv.imshow('perceptual-reflection-removal Reflection', output_image_r)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()