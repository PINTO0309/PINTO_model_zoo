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
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    result = np.array(result)

    output_image1 = result[0][0]
    output_image1 = output_image1.transpose(1, 2, 0)
    output_image1 = np.clip(output_image1 * 255.0, 0, 255).astype(np.uint8)
    output_image1 = cv.cvtColor(output_image1, cv.COLOR_RGB2BGR)

    output_image2 = result[1][0]
    output_image2 = output_image2.transpose(1, 2, 0)
    output_image2 = np.clip(output_image2 * 255.0, 0, 255).astype(np.uint8)
    output_image2 = cv.cvtColor(output_image2, cv.COLOR_RGB2BGR)

    return output_image1, output_image2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=
        'hinet_derain_test100_256x320/hinet_derain_test100_256x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,320',
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
    onnx_session = onnxruntime.InferenceSession(model_path)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image1, output_image2 = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        output_image1 = cv.resize(output_image1,
                                  dsize=(frame_width, frame_height))
        output_image2 = cv.resize(output_image2,
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
        cv.imshow('HINet Input', debug_image)
        cv.imshow('HINet Output1', output_image1)
        cv.imshow('HINet Output2', output_image2)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()