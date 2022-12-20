#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, fused_argmax_scale_ratio):
    # ONNX Infomation
    input_name01 = onnx_session.get_inputs()[0].name
    input_name02 = onnx_session.get_inputs()[1].name
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, expand dimensions, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255.0 - mean) / std
    input_image = input_image.transpose(2, 0, 1).astype('float32')
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    result = onnx_session.run(
        None,
        {
            input_name01: input_image,
            input_name02: np.array(
                (fused_argmax_scale_ratio)).astype(np.float32),
        },
    )

    # Post process:squeeze, Transpose, Resize, argmax
    segmentation_map = result[0]
    segmentation_map = np.squeeze(segmentation_map)

    return segmentation_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='pidnet_S_cityscapes_544x960_fused_argmax.onnx',
    )
    parser.add_argument("--fused_argmax_scale_ratio", type=float, default=1.0)

    args = parser.parse_args()

    model_path = args.model

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    fused_argmax_scale_ratio = args.fused_argmax_scale_ratio

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            # 'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        segmentation_map = run_inference(
            onnx_session,
            frame,
            fused_argmax_scale_ratio,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            segmentation_map,
            class_num=19,  # CityScapes:19, CamVid:11
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('PIDNet Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3 + 2] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def draw_debug(image, elapsed_time, segmentation_map, class_num=19):
    # color map list
    color_map = get_color_map_list(class_num)

    # Overlay segmentation map
    for index in range(0, class_num):
        mask = np.where(segmentation_map == index, 0, 1)

        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (color_map[index * 3 + 0], color_map[index * 3 + 1],
                       color_map[index * 3 + 2])

        # Overlay
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask = cv.resize(mask, (image.shape[1], image.shape[0]))
        mask_image = np.where(mask, image, bg_image)
        image = cv.addWeighted(image, 0.5, mask_image, 0.5, 1.0)

    # Inference elapsed time
    cv.putText(image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
