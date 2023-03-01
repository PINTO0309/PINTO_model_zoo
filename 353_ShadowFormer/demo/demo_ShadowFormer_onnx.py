#!/usr/bin/env python
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, mask_image):
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, transpose, normarize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    mask_input_image = cv.resize(mask_image, dsize=(input_width, input_height))
    mask_input_image = mask_input_image / 255.0
    mask_input_image = np.expand_dims(mask_input_image, axis=0)
    mask_input_image = np.expand_dims(mask_input_image, axis=0)
    mask_input_image = mask_input_image.astype('float32')

    # Inference
    input_name_01 = onnx_session.get_inputs()[0].name
    input_name_02 = onnx_session.get_inputs()[1].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {
        input_name_01: input_image,
        input_name_02: mask_input_image
    })

    # # Post process:squeeze, uint8 cast, RGB->BGR,
    output_image = result[0]
    output_image = np.squeeze(output_image)
    output_image = output_image.transpose(1, 2, 0)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    global mouse_point, mouse_event

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        default='sample.jpg',
    )
    parser.add_argument(
        '--mask',
        type=str,
        default='sample_mask.jpg',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='shadowformer_istd_plus_480x640.onnx',
    )
    args = parser.parse_args()
    model_path = args.model
    image_path = args.image
    mask_path = args.mask

    # Load Image
    image = cv.imread(image_path)
    mask_image = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    start_time = time.time()

    # Inference execution
    output_image = run_inference(
        onnx_session,
        image,
        mask_image,
    )

    elapsed_time = time.time() - start_time

    # Draw
    debug_image = copy.deepcopy(image)
    cv.putText(debug_image,
               'Elapsed Time : ' + '{:.1f}'.format(elapsed_time * 1000) + 'ms',
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
               cv.LINE_AA)

    cv.imshow('ShadowFormer : Input', debug_image)
    cv.imshow('ShadowFormer : Mask', mask_image)
    cv.imshow('ShadowFormer : Output', output_image)
    _ = cv.waitKey(-1)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
