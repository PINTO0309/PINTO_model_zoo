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
        "--model",
        type=str,
        default='saved_model_256x256/cocosnet.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
    parser.add_argument(
        "--input_seg_map",
        type=str,
        default='image/input_seg_map/2.png',
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default='image/ref_image/1.jpg',
    )
    parser.add_argument(
        "--ref_seg_map",
        type=str,
        default='image/ref_seg_map/1.png',
    )

    args = parser.parse_args()

    return args


def run_inference(
    onnx_session,
    input_size,
    input_seg_map,
    ref_image,
    ref_seg_map,
):
    # input_seg_map
    # Pre process:One hot label for each channel
    resize_input_seg = cv.resize(input_seg_map, (input_size[1], input_size[0]))

    input_seg_map_list = np.zeros((151, *input_size), dtype=np.float32)
    for index in range(1, len(input_seg_map_list)):
        input_seg_map_list[index - 1] = np.where(resize_input_seg == index, 1,
                                                 0)
    input_seg_map_list = np.expand_dims(input_seg_map_list, axis=0)

    # ref_image
    # Pre process:Resize, Transpose, float32 cast, 1/255.0
    resize_ref_image = cv.resize(ref_image, (input_size[1], input_size[0]))
    resize_ref_image = resize_ref_image.transpose(2, 0, 1)
    resize_ref_image = np.expand_dims(resize_ref_image, axis=0)
    resize_ref_image = resize_ref_image.astype('float32')
    resize_ref_image /= 255

    # ref_seg_map
    # Pre process:One hot label for each channel
    resize_ref_seg = cv.resize(ref_seg_map, (input_size[1], input_size[0]))

    ref_seg_map_list = np.zeros((151, *input_size), dtype=np.float32)
    for index in range(1, len(ref_seg_map_list)):
        ref_seg_map_list[index - 1] = np.where(resize_ref_seg == index, 1, 0)
    ref_seg_map_list = np.expand_dims(ref_seg_map_list, axis=0)

    # Inference
    input_name_01 = onnx_session.get_inputs()[0].name
    input_name_02 = onnx_session.get_inputs()[1].name
    input_name_03 = onnx_session.get_inputs()[2].name
    output_name = onnx_session.get_outputs()[0].name

    result = onnx_session.run(
        [output_name],
        {
            input_name_01: input_seg_map_list,
            input_name_02: resize_ref_image,
            input_name_03: ref_seg_map_list
        },
    )

    # Post process:Squeeze, Transpose
    output_image = np.squeeze(result)
    output_image = output_image.transpose(1, 2, 0)

    return output_image


def main():
    args = get_args()

    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    input_seg_map_path = args.input_seg_map
    ref_image_path = args.ref_image
    ref_seg_map_path = args.ref_seg_map

    # Read Image
    input_seg_map = cv.imread(input_seg_map_path, cv.IMREAD_GRAYSCALE)
    ref_image = cv.imread(ref_image_path)
    ref_seg_map = cv.imread(ref_seg_map_path, cv.IMREAD_GRAYSCALE)

    # Load model
    print('Load model')
    onnx_session = onnxruntime.InferenceSession(model_path)

    # Inference execution
    start_time = time.time()

    print('Run inference')
    output_image = run_inference(
        onnx_session,
        input_size,
        input_seg_map,
        ref_image,
        ref_seg_map,
    )

    elapsed_time = time.time() - start_time
    print('Elapsed Time:', elapsed_time)

    input_seg_map, ref_image, ref_seg_map, output_image = draw_image(
        input_size,
        input_seg_map,
        ref_image,
        ref_seg_map,
        output_image,
    )

    cv.imshow('input_seg_map', input_seg_map)
    cv.imshow('ref_image', ref_image)
    cv.imshow('ref_seg_map', ref_seg_map)
    cv.imshow('output', output_image)
    cv.waitKey(0)


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = [(37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255]
    return np.array(color, dtype=np.uint8)


def draw_image(
    input_size,
    input_seg_map,
    ref_image,
    ref_seg_map,
    output_image,
):
    # Resize
    input_seg_map = cv.resize(input_seg_map,
                              dsize=(input_size[0], input_size[1]))
    ref_image = cv.resize(ref_image, dsize=(input_size[0], input_size[1]))
    ref_seg_map = cv.resize(ref_seg_map, dsize=(input_size[0], input_size[1]))

    # input_seg_map
    input_seg_map_color = np.zeros((*input_size, 3), dtype=np.uint8)

    for index in range(255):
        mask = np.where(input_seg_map == index, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')

        color_image = np.zeros((*input_size, 3), dtype=np.uint8)
        color_image += get_id_color(index)

        input_seg_map_color = np.where(mask, input_seg_map_color, color_image)

    # ref_seg_map
    ref_seg_map_color = np.zeros((*input_size, 3), dtype=np.uint8)

    for index in range(255):
        mask = np.where(ref_seg_map == index, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')

        color_image = np.zeros((*input_size, 3), dtype=np.uint8)
        color_image += get_id_color(index)

        ref_seg_map_color = np.where(mask, ref_seg_map_color, color_image)

    return input_seg_map_color, ref_image, ref_seg_map_color, output_image


if __name__ == '__main__':
    main()
