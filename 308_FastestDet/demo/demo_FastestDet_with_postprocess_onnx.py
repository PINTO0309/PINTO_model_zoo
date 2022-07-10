#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image, score_th=0.8):
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    results = np.squeeze(results)
    bboxes, scores, class_ids = [], [], []
    for result in results:
        bbox = result[:4].tolist()
        score = result[5]
        class_id = int(result[4])

        if score_th > score:
            continue

        bboxes.append(bbox)
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
        default='fastestdet_post_352x352.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='352,352',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        bboxes, scores, class_ids = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        # Draw
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            y1, x1 = int(bbox[0] * frame_height), int(bbox[1] * frame_width)
            y2, x2 = int(bbox[2] * frame_height), int(bbox[3] * frame_width)

            cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv.putText(debug_image, '%d:%.2f' % (class_id, score),
                       (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('FastestDet', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()