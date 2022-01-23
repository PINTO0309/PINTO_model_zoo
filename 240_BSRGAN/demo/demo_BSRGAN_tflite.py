#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    hr_image = interpreter.get_tensor(output_details[0]['index'])

    # Post process:squeeze, Transpoes, uint8 cast, RGB->BGR,
    hr_image = np.squeeze(hr_image)
    hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
    hr_image = cv.cvtColor(hr_image, cv.COLOR_RGB2BGR)

    return hr_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='bsrganx2_128x128/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]
    input_width, input_height = input_size[1], input_size[0]

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        debug_image = cv.resize(debug_image, dsize=(input_width, input_height))

        # Inference execution
        hr_image = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        original_image, concat_image, _, _ = draw_debug(
            debug_image,
            elapsed_time,
            hr_image,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('BSRGAN Demo : Original', original_image)
        cv.imshow('BSRGAN Demo : HR', concat_image)

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
    cv.putText(debug_image, "Right image : BSRGAN", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    concat_image = cv.hconcat([debug_image, hr_image])

    return image, concat_image, debug_image, hr_image


if __name__ == '__main__':
    main()