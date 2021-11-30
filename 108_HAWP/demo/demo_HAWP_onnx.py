#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Standardization,Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255 - mean) / std

    input_image = input_image.transpose(2, 0, 1).astype('float32')
    input_image = input_image.reshape(-1, 3, input_size[1], input_size[0])

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    lines, scores = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    image_width, image_height = image.shape[1], image.shape[0]
    for line in lines:
        line[0] = int(line[0] / 128 * image_width)
        line[1] = int(line[1] / 128 * image_height)
        line[2] = int(line[2] / 128 * image_width)
        line[3] = int(line[3] / 128 * image_height)

    return lines, scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='hawp_512x512_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,512',
    )
    parser.add_argument("--score_th", type=str, default=0.95)

    args = parser.parse_args()
    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]
    score_th = args.score_th

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    start_time = time.time()

    # Read Sample Image
    image = cv.imread('sample.png')
    debug_image = copy.deepcopy(image)

    # Inference execution
    lines, scores = run_inference(
        onnx_session,
        input_size,
        image,
    )

    elapsed_time = time.time() - start_time

    # Draw Line
    for line, score in zip(lines, scores):
        if score < score_th:
            continue
        x1, y1 = int(line[0]), int(line[1])
        x2, y2 = int(line[2]), int(line[3])
        cv.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
               cv.LINE_AA)

    cv.imshow('HAWP Input', debug_image)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
