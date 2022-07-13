#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image, score_th):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    input_image = (input_image / 255 - mean) / std
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    result = np.array(result)
    output_image = np.squeeze(result)
    if score_th is not None:
        output_image = np.where(output_image > score_th, 1, 0)
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.png')
    parser.add_argument("--score_th", type=float, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='imageforensicsosn_480x640/imageforensicsosn_480x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size
    image_path = args.image
    score_th = args.score_th

    input_size = [int(i) for i in input_size.split(',')]

    # Read Image
    image = cv.imread(image_path)

    debug_image = copy.deepcopy(image)
    image_height, image_width = image.shape[0], image.shape[1]

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # Inference execution
    output_image = run_inference(
        onnx_session,
        input_size,
        image,
        score_th,
    )

    output_image = cv.resize(output_image, dsize=(image_width, image_height))

    cv.imshow('ImageForensicsOSN Input', debug_image)
    cv.imshow('ImageForensicsOSN Output', output_image)

    _ = cv.waitKey(-1)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()