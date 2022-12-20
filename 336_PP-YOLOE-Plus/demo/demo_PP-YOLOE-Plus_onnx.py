#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, scale_factor, score_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    # ONNX Infomation
    input_name01 = onnx_session.get_inputs()[0].name
    input_name02 = onnx_session.get_inputs()[1].name
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    input_scale_factor = np.array([[scale_factor,
                                    scale_factor]]).astype(np.float32)

    # Inference
    results = onnx_session.run(
        None,
        {
            input_name01: input_image,
            input_name02: input_scale_factor
        },
    )

    # Post process
    results = np.squeeze(results[0])
    bboxes, scores, class_ids = [], [], []
    for result in results:
        bbox = result[2:].tolist()
        score = result[1]
        class_id = int(result[0])

        if score_th > score:
            continue

        x1 = int((bbox[0] / input_width) * image_width)
        y1 = int((bbox[1] / input_height) * image_height)
        x2 = int((bbox[2] / input_width) * image_width)
        y2 = int((bbox[3] / input_height) * image_height)
        bboxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
        scores.append(score)

    return bboxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='ppyoloe_plus_crn_s_80e_coco_640x640.onnx',
    )
    parser.add_argument("--scale_factor", type=float, default=1.0)

    args = parser.parse_args()
    model_path = args.model
    scale_factor = args.scale_factor

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
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
        bboxes, scores, class_ids = run_inference(
            onnx_session,
            frame,
            scale_factor,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            bboxes,
            scores,
            class_ids,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('PP-YOLOE-Plus', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    debug_image,
    elapsed_time,
    bboxes,
    scores,
    class_ids,
):
    # Object Detection
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])

        cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv.putText(debug_image, '%d:%.2f' % (class_id, score), (x1, y1 - 5), 0,
                   0.7, (255, 255, 0), 2)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
