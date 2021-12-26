#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    original_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    original_image = original_image / 255.0

    # Pre process:Resize, BGR->RGB, float32 cast
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
    A = interpreter.get_tensor(output_details[0]['index'])

    # Post process:squeeze, RGB->BGR, uint8 cast
    A = np.array(A)
    r1 = A[:, :, :, :3]
    r2 = A[:, :, :, 3:6]
    r3 = A[:, :, :, 6:9]
    r4 = A[:, :, :, 9:12]
    r5 = A[:, :, :, 12:15]
    r6 = A[:, :, :, 15:18]
    r7 = A[:, :, :, 18:21]
    r8 = A[:, :, :, 21:24]
    x = original_image + r1 * (np.power(original_image, 2) - original_image)
    x = x + r2 * (np.power(x, 2) - x)
    x = x + r3 * (np.power(x, 2) - x)
    enhanced_image_1 = x + r4 * (np.power(x, 2) - x)
    x = enhanced_image_1 + r5 * (np.power(enhanced_image_1, 2) -
                                 enhanced_image_1)
    x = x + r6 * (np.power(x, 2) - x)
    x = x + r7 * (np.power(x, 2) - x)
    output_image = x + r8 * (np.power(x, 2) - x)
    output_image = output_image[0, :, :, :]
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float16_quant.tflite',
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
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

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
            interpreter,
            input_size,
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
        cv.imshow('AOD-Net Input', debug_image)
        cv.imshow('AOD-Net Output', output_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()