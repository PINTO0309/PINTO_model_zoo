#!/usr/bin/env python

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(
    onnx_session,
    input_name,
    input_size,
    image,
    score_th=0.8,
):
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...]
    input_image = (input_image - 127.5) / 127.5
    input_image = input_image.astype('float32')
    # Inference
    results = onnx_session.run(None, {input_name: input_image})[0]
    # Post process
    bboxes, scores, class_ids = [], [], []
    for batchno_classid_score_x1y1x2y2 in results:
        bbox = batchno_classid_score_x1y1x2y2[-4:].tolist()
        class_id = int(batchno_classid_score_x1y1x2y2[1])
        score = batchno_classid_score_x1y1x2y2[2]
        if score_th > score:
            continue
        image_height, image_width = image.shape[0], image.shape[1]
        bbox[0] = int(bbox[0] * image_width)
        bbox[1] = int(bbox[1] * image_height)
        bbox[2] = int(bbox[2] * image_width)
        bbox[3] = int(bbox[3] * image_height)
        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
    return bboxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-mov',
        '--movie',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-mod',
        '--model',
        type=str,
        default='RFB-320_240x320_post.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    parser.add_argument(
        '-s',
        '--score_th',
        type=float,
        default=0.7,
    )
    args = parser.parse_args()
    model_path = args.model
    provider = args.provider
    score_th = args.score_th

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
    providers = []
    if provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif provider == 'tensorrt':
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
        path_or_bytes=model_path,
        providers=providers,
    )
    input_name = onnx_session.get_inputs()[0].name
    input_size = onnx_session.get_inputs()[0].shape

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
            input_name,
            input_size,
            frame,
            score_th=score_th,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            f'Elapsed Time: {elapsed_time * 1000:.1f} ms',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        # Draw
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            y1, x1 = int(bbox[1]), int(bbox[0])
            y2, x2 = int(bbox[3]), int(bbox[2])
            cv.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )
            cv.putText(
                debug_image,
                f'{class_id:d}:{score:.2f}',
                (x1, y1 - 5),
                0,
                0.7,
                (0, 255, 0),
                2,
            )

        video_writer.write(debug_image)
        cv.imshow('RFB320 ONNX', debug_image)
        key = cv.waitKey(1) if args.movie is None or args.movie[-4:] == '.mp4' else cv.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()