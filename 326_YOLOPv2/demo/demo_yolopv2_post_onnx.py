#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image, score_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

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
    drivable_area = np.squeeze(results[3])
    lane_line = np.squeeze(results[4])
    scores = results[5]
    batchno_classid_y1x1y2x2 = results[6]

    # Drivable Area Segmentation
    drivable_area = drivable_area.transpose(1, 2, 0)
    drivable_area = cv.resize(
        drivable_area,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )
    drivable_area = drivable_area.transpose(2, 0, 1)

    # Lane Line
    lane_line = cv.resize(
        lane_line,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )

    # Traffic Object Detection
    od_bboxes, od_scores, od_class_ids = [], [], []
    for score, batchno_classid_y1x1y2x2_ in zip(
            scores,
            batchno_classid_y1x1y2x2,
    ):
        class_id = int(batchno_classid_y1x1y2x2_[1])

        if score_th > score:
            continue

        y1 = batchno_classid_y1x1y2x2_[-4]
        x1 = batchno_classid_y1x1y2x2_[-3]
        y2 = batchno_classid_y1x1y2x2_[-2]
        x2 = batchno_classid_y1x1y2x2_[-1]
        y1 = int(y1 * (image_height / input_size[0]))
        x1 = int(x1 * (image_width / input_size[1]))
        y2 = int(y2 * (image_height / input_size[0]))
        x2 = int(x2 * (image_width / input_size[1]))

        od_bboxes.append([x1, y1, x2, y2])
        od_class_ids.append(class_id)
        od_scores.append(score)

    return drivable_area, lane_line, od_bboxes, od_scores, od_class_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='yolopv2_post_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
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
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        drivable_area, lane_line, bboxes, scores, class_ids = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            drivable_area,
            lane_line,
            bboxes,
            scores,
            class_ids,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('YOLOP v2', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    debug_image,
    elapsed_time,
    drivable_area,
    lane_line,
    bboxes,
    scores,
    class_ids,
):
    # Draw:Drivable Area Segmentation
    # Not in Drivable Area
    bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = (255, 0, 0)

    mask = np.where(drivable_area[0] > 0.5, 0, 1)
    mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
    mask_image = np.where(mask, debug_image, bg_image)
    debug_image = cv.addWeighted(debug_image, 0.75, mask_image, 0.25, 1.0)

    # Drivable Area
    bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = (0, 255, 0)

    mask = np.where(drivable_area[1] > 0.5, 0, 1)
    mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
    mask_image = np.where(mask, debug_image, bg_image)
    debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # Draw:Lane Line
    bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = (0, 0, 255)

    mask = np.where(lane_line > 0.5, 0, 1)
    mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
    mask_image = np.where(mask, debug_image, bg_image)
    debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # Draw:Traffic Object Detection
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
