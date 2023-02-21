#!/usr/bin/env python
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

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze, uint8 cast, RGB->BGR,
    hr_image = result[0]
    hr_image = np.squeeze(hr_image)
    hr_image = hr_image.transpose(1, 2, 0)
    hr_image = np.clip(hr_image, 0, 255).astype(np.uint8)
    hr_image = cv.cvtColor(hr_image, cv.COLOR_RGB2BGR)

    return hr_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='rfdn_120x160.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Resize for Demo
        debug_image = cv.resize(
            debug_image,
            dsize=(input_width, input_height),
            interpolation=cv.INTER_LINEAR,
        )

        # Inference execution
        hr_image = run_inference(
            onnx_session,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        original_image, _, debug_image, hr_image = draw_debug(
            debug_image,
            elapsed_time,
            hr_image,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('RFDN Demo : Original', original_image)
        cv.imshow('RFDN Demo : Input', debug_image)
        cv.imshow('RFDN Demo : Output', hr_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, hr_image):
    hr_width, hr_height = hr_image.shape[1], hr_image.shape[0]

    # Up-conversion using OpenCV Resize for comparison
    debug_image = copy.deepcopy(image)
    debug_image = cv.resize(
        debug_image,
        dsize=(hr_width, hr_height),
        interpolation=cv.INTER_LINEAR,
    )

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)
    cv.putText(debug_image, "Left image : Bilinear interpolation", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(debug_image, "Right image : RFDN", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(
        debug_image,
        str(image.shape[1]) + 'x' + str(image.shape[0]) + ' -> ' +
        str(hr_image.shape[1]) + 'x' + str(hr_image.shape[0]), (10, 120),
        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    concat_image = cv.hconcat([debug_image, hr_image])

    return image, concat_image, debug_image, hr_image


if __name__ == '__main__':
    main()