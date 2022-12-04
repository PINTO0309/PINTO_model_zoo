#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

MEAN = np.asarray([103.94, 116.78, 123.68], dtype=np.float32).reshape([1,3,1,1])
STD = np.asarray([57.38, 57.12, 58.40], dtype=np.float32).reshape([1,3,1,1])

def run_inference(onnx_session, input_shape, image, score_th):
    # Pre process: Creates 4-dimensional blob from image
    size = (input_shape[3], input_shape[2])
    input_image = cv.dnn.blobFromImage(image, size=size, swapRB=True)
    input_image = (input_image - MEAN) / STD

    # Inference
    input_name = 'input'
    output_names = ['x1y1x2y2_score_class', 'final_masks']
    results = onnx_session.run(output_names, {input_name: input_image})

    def crop(bbox, shape):
        x1 = int(max(bbox[0] * shape[1], 0))
        y1 = int(max(bbox[1] * shape[0], 0))
        x2 = int(max(bbox[2] * shape[1], 0))
        y2 = int(max(bbox[3] * shape[0], 0))
        return (slice(y1, y2), slice(x1, x2))

    # Post process
    bboxes, scores, class_ids, masks = [], [], [], []
    for result, mask in zip(results[0][0], results[1]):
        bbox = result[:4].tolist()
        score = result[4]
        class_id = int(result[5])

        if score_th > score:
            continue

        # Add 1 to class_id to distinguish it from the background 0
        mask = np.where(mask > 0.5, class_id + 1, 0).astype(np.uint8)
        region = crop(bbox, mask.shape)
        cropped = np.zeros(mask.shape, dtype=np.uint8)
        cropped[region] = mask[region]

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
        masks.append(cropped)

    return bboxes, scores, class_ids, masks

def get_colors(num):
    colors = [[0, 0, 0]]
    np.random.seed(0)
    for i in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str, default='yolact_im700_54_800000_700x700_post.onnx')
    parser.add_argument("-trt", "--use_tensorrt", action='store_true')
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-mv", "--movie", type=str, default=None)
    parser.add_argument("-im", "--image", type=str, default=None)
    parser.add_argument("-th", "--threshold", type=float, default=0.1)
    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    elif args.image is not None:
        cap_device = args.image
    cap = cv.VideoCapture(cap_device)

    # Load model
    providers = []
    if args.use_tensorrt:
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    else:
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
    )

    # Prepare color table
    colors = get_colors(len(COCO_CLASSES))
    input_shape = onnx_session.get_inputs()[0].shape
    output_mask = onnx_session.get_outputs()[1]
    mask_shape = np.asarray([output_mask.shape[1]]+[output_mask.shape[2]]+[3], dtype=np.int32)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        bboxes, scores, class_ids, masks = run_inference(
            onnx_session,
            input_shape,
            frame,
            score_th=args.threshold
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            f"Elapsed Time : {elapsed_time * 1000:.1f}ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        # Draw
        if len(masks) > 0:
            mask_image = np.zeros(mask_shape, dtype=np.uint8)
            for mask in masks:
                color_mask = np.array(colors, dtype=np.uint8)[mask]
                filled = np.nonzero(mask)
                mask_image[filled] = color_mask[filled]
            mask_image = cv.resize(mask_image, (frame_width, frame_height), cv.INTER_NEAREST)
            cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 0.0, debug_image)

        for bbox, score, class_id, mask in zip(bboxes, scores, class_ids, masks):
            x1, y1 = int(bbox[0] * frame_width), int(bbox[1] * frame_height)
            x2, y2 = int(bbox[2] * frame_width), int(bbox[3] * frame_height)
            cv.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )
            cv.putText(
                debug_image,
                f'{COCO_CLASSES[class_id]}:{score:.2f}',
                (x1, y1 - 5),
                0,
                0.7,
                (0, 255, 0),
                2,
            )

        cv.imshow('Yolact', debug_image)

        key = cv.waitKey(0 if args.image is not None else 1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
