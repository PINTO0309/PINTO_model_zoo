#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from typing import Any, Tuple

import cv2
import numpy as np
import onnxruntime  # type: ignore


def minmax_scale(input_array: np.ndarray) -> np.ndarray:
    min_val: float = np.min(input_array)
    max_val: float = np.max(input_array)

    output_array: np.ndarray = (input_array - min_val) * 255.0 / (max_val -
                                                                  min_val)

    return output_array


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> np.ndarray:
    input_detail = onnx_session.get_inputs()[0]
    input_name: str = input_detail.name
    input_shape: Tuple[int, int] = input_detail.shape[2:4]

    # Pre process: Resize, Normalize, Transpose, Expand Dims, float32 cast
    input_image: np.ndarray = cv2.resize(
        image,
        dsize=(input_shape[1], input_shape[0]),
    )
    input_image = np.divide(np.array(input_image, np.float32), 127.5) - 1.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    result: Any = onnx_session.run(None, {input_name: input_image})

    # Post process: squeeze, MinMax Scale, uint8 cast
    output_image: np.ndarray = np.squeeze(result[0])
    for i in range(output_image.shape[2]):
        output_image[:, :, i] = minmax_scale(output_image[:, :, i])
    output_image = output_image.astype(np.uint8)

    return output_image


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.png')
    parser.add_argument(
        "--model",
        type=str,
        default='attentive_gan_derainnet_240x360/model_float32.onnx',
    )

    args = parser.parse_args()
    model_path: str = args.model
    image_path: str = args.image

    # Load model
    onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    image: np.ndarray = cv2.imread(image_path)
    debug_image: np.ndarray = copy.deepcopy(image)
    image_height: int = image.shape[0]
    image_width: int = image.shape[1]

    start_time: float = time.time()

    # Inference execution
    output_image: np.ndarray = run_inference(
        onnx_session,
        image,
    )

    output_image = cv2.resize(output_image, dsize=(image_width, image_height))

    elapsed_time: float = time.time() - start_time

    # Inference elapsed time
    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Display
    cv2.imshow('attentive-gan-derainnet Input', debug_image)
    cv2.imshow('attentive-gan-derainnet Output', output_image)
    _ = cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
