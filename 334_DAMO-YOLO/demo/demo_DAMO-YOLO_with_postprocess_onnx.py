#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, score_th=0.8):
    # ONNX Infomation
    input_name = onnx_session.get_inputs()[0].name
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    bboxes, scores, class_ids = [], [], []
    for score, batchno_classid_x1y1x2y2 in zip(results[0], results[1]):
        bbox = batchno_classid_x1y1x2y2[-4:].tolist()
        class_id = int(batchno_classid_x1y1x2y2[1])
        score = score[0]

        if score_th > score:
            continue

        image_height, image_width = image.shape[0], image.shape[1]
        bbox[0] = int((bbox[0] / input_width) * image_width)
        bbox[1] = int((bbox[1] / input_height) * image_height)
        bbox[2] = int((bbox[2] / input_width) * image_width)
        bbox[3] = int((bbox[3] / input_height) * image_height)

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
        default='damoyolo_tinynasL20_T_192x320_post.onnx',
    )
    parser.add_argument("--score_th", type=float, default=0.4)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        '--use_cuda',
        action='store_true',
    )
    device_group.add_argument(
        '--use_tensorrt',
        action='store_true',
    )

    args = parser.parse_args()
    model_path = args.model
    score_th = args.score_th
    use_cuda = args.use_cuda
    use_tensorrt = args.use_tensorrt

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # Load model
    providers = None
    if not use_cuda and not use_tensorrt:
        providers = [
            'CPUExecutionProvider',
        ]
    elif use_cuda:
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif use_tensorrt:
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
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
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
            score_th=score_th,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        # Draw
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            y1, x1 = int(bbox[1]), int(bbox[0])
            y2, x2 = int(bbox[3]), int(bbox[2])

            cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv.putText(debug_image, '%d:%.2f' % (class_id, score),
                       (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)

        video_writer.write(debug_image)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('DAMO-YOLO ONNX', debug_image)

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
