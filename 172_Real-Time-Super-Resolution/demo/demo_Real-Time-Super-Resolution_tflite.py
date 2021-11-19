#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    # Pre process:Resize, BGR->RGB, expand dimensions, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    sr_image = interpreter.get_tensor(output_details[0]['index'])

    # Post process:squeeze value clip, uint8 cast, RGB->BGR,
    sr_image = np.squeeze(sr_image)
    sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)
    sr_image = cv.cvtColor(sr_image, cv.COLOR_RGB2BGR)

    return sr_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_96x96/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='96,96',
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
        sr_image = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        original_image, concat_image, _, _ = draw_debug(
            debug_image,
            elapsed_time,
            sr_image,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('Real-Time-Super-Resolution Demo : Original', original_image)
        cv.imshow('Real-Time-Super-Resolution Demo : SR', concat_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, sr_image):
    sr_width, sr_height = sr_image.shape[1], sr_image.shape[0]

    # Up-conversion using OpenCV Resize for comparison
    debug_image = copy.deepcopy(image)
    debug_image = cv.resize(
        debug_image,
        dsize=(sr_width, sr_height),
        interpolation=cv.INTER_LINEAR,
    )

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)
    cv.putText(debug_image, "Left : Bilinear interpolation", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(debug_image, "Right : Super Resolution", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    concat_image = cv.hconcat([debug_image, sr_image])

    return image, concat_image, debug_image, sr_image


if __name__ == '__main__':
    main()