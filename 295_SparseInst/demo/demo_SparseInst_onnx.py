#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image, score_th):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process
    mask_preds = np.array(result[0])
    scores = np.array(result[1])
    labels = np.array(result[2])

    # Extraction by score threshold
    mask_preds = mask_preds[scores > score_th]
    labels = labels[scores > score_th]
    scores = scores[scores > score_th]

    return mask_preds, scores, labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--score_th", type=float, default=0.6)
    parser.add_argument(
        "--model",
        type=str,
        default=
        'sparseinst_r50_giam_aug_768x1344/sparseinst_r50_giam_aug_768x1344_opt.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='768,1344',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size
    score_th = args.score_th

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # Inference execution
        masks, scores, labels = run_inference(
            onnx_session,
            input_size,
            frame,
            score_th,
        )

        # Bounding Rectangle
        bboxes_list = []
        for mask in masks:
            contours, _ = cv.findContours((mask * 255).astype('uint8'),
                                          cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
            bbox = []
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                x = int((x / input_size[1]) * frame_width)
                y = int((y / input_size[0]) * frame_height)
                w = int((w / input_size[1]) * frame_width)
                h = int((h / input_size[0]) * frame_height)
                bbox.append([x, y, x + w, y + h])
            bboxes_list.append(bbox)

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            masks,
            scores,
            labels,
            bboxes_list,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('SparseInst Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(image, elapsed_time, score_th, masks, scores, labels,
               bboxes_list):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for index, (mask, score, label,
                bboxes) in enumerate(zip(masks, scores, labels, bboxes_list)):
        if score < score_th:
            continue

        color = get_id_color(index)

        # Color image
        color_image = np.zeros(image.shape, dtype=np.uint8)
        color_image[:] = color

        # Resized mask image
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        resize_mask = cv.resize(mask, (image_width, image_height))

        # Mask addWeighted
        mask_image = np.where(resize_mask, color_image, debug_image)
        debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

        # Bounding box & Lable
        for bbox in bboxes:
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0))
            cv.putText(debug_image, str(label), (bbox[0], bbox[1] + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                       cv.LINE_AA)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
