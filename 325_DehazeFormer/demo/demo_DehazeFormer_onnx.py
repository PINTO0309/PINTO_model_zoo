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

    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0
    input_image = input_image * 2 - 1

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    result = np.array(result)
    output_image = np.squeeze(result)
    output_image = output_image.transpose(1, 2, 0)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='dehazeformer_t_outdoor_360x640.onnx',
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
        output_image = run_inference(
            onnx_session,
            frame,
        )

        output_image = cv.resize(output_image,
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
        cv.imshow('DehazeFormer Input', debug_image)
        cv.imshow('DehazeFormer Output', output_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
