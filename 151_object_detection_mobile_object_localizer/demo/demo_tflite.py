#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    # Pre process:Resize, BGR->RGB, Reshape, float32 cast
    input_image = cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.float32)

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    output_details = interpreter.get_output_details()

    # Post process:Extract each information
    bboxes = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[3]['index'])
    num = interpreter.get_tensor(output_details[1]['index'])

    bboxes = np.squeeze(bboxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    num = int(num[0])

    return bboxes, classes, scores, num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float16_quant.tflite',
    )
    parser.add_argument("--score", type=float, default=0.2)

    args = parser.parse_args()

    model_path = args.model
    score_th = args.score

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_size = 192

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        bboxes, classes, scores, num = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            num,
            bboxes,
            classes,
            scores,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('Mobile Object Localizer Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    num,
    bboxes,
    classes,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for i in range(num):
        score = scores[i]
        bbox = bboxes[i]
        # class_id = classes[i].astype(np.int) + 1

        if score < score_th:
            continue

        # Bounding box
        x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
        x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)

        cv.putText(debug_image, '{:.3f}'.format(score), (x1, y1 - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()