#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->YCR_CB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2YCR_CB)
    input_image = input_image / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    result = np.array(result)

    output_image = result[0][0]
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_YCrCb2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='image/sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_batch_1/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,672',
    )

    args = parser.parse_args()

    imag_path = args.image
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    start_time = time.time()

    # Read Image
    frame = cv.imread(imag_path)
    debug_image = copy.deepcopy(frame)
    frame_height, frame_width = frame.shape[0], frame.shape[1]

    # Inference execution
    output_image = run_inference(
        onnx_session,
        input_size,
        frame,
    )

    output_image = cv.resize(output_image, dsize=(frame_width, frame_height))
    elapsed_time = time.time() - start_time

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
               cv.LINE_AA)

    cv.imshow('Single-Image-Desnowing-HDCWNet Input', debug_image)
    cv.imshow('Single-Image-Desnowing-HDCWNet Output', output_image)

    key = cv.waitKey(-1)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()