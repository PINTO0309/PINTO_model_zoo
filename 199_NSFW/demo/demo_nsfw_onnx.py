#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default='image/sample.jpg',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_nsfw/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='224,224',
    )

    args = parser.parse_args()

    return args


def run_inference(
    onnx_session,
    input_size,
    image,
):
    # Pre process:Resize, RGB->BGR, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    vgg_mean = [104, 117, 123]
    input_image = input_image - vgg_mean

    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run(
        [output_name],
        {input_name: input_image},
    )

    # Post process
    onnx_result = np.squeeze(onnx_result).astype(np.float32)

    return onnx_result


def main():
    args = get_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    # read image
    image_path = args.image
    image = cv.imread(image_path)

    # Inference execution
    start_time = time.time()
    result = run_inference(
        onnx_session,
        input_size,
        image,
    )
    elapsed_time = time.time() - start_time

    print('Elapsed Time :', '{:.1f}'.format(elapsed_time * 1000) + "ms")
    print('sfw :', '{:.3f}'.format(result[0]))
    print('nsfw:', '{:.3f}'.format(result[1]))


if __name__ == '__main__':
    main()
