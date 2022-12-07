#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime


class DAMOYOLO(object):
    def __init__(
        self,
        model_path,
        max_num=500,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.max_num = max_num

        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name

        self.input_shape = self.input_detail.shape[2:]

    def __call__(self, image, score_th=0.05, nms_th=0.8):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        image, ratio = self._preprocess(temp_image, self.input_shape)

        results = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )

        scores = results[0]
        bboxes = results[1]
        bboxes, scores, class_ids = self._postprocess(
            scores,
            bboxes,
            score_th,
            nms_th,
        )

        decode_ratio = min(image_height / int(image_height * ratio),
                           image_width / int(image_width * ratio))
        if len(bboxes) > 0:
            bboxes = bboxes * decode_ratio

        return bboxes, scores, class_ids

    def _preprocess(self, image, input_size, swap=(2, 0, 1)):
        temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 3:
            padded_image = np.ones((input_size[0], input_size[1], 3),
                                   dtype=np.uint8)
        else:
            padded_image = np.ones(input_size, dtype=np.uint8)

        ratio = min(input_size[0] / temp_image.shape[0],
                    input_size[1] / temp_image.shape[1])
        resized_image = cv2.resize(
            temp_image,
            (int(temp_image.shape[1] * ratio), int(
                temp_image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[:int(temp_image.shape[0] *
                          ratio), :int(temp_image.shape[1] *
                                       ratio)] = resized_image
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        scores,
        bboxes,
        score_th,
        nms_th,
    ):
        batch_size = bboxes.shape[0]
        for i in range(batch_size):
            if not bboxes[i].shape[0]:
                continue
            bboxes, scores, class_ids = self._multiclass_nms(
                bboxes[i],
                scores[i],
                score_th,
                nms_th,
                self.max_num,
            )

        return bboxes, scores, class_ids

    def _multiclass_nms(
        self,
        bboxes,
        scores,
        score_th,
        nms_th,
        max_num=100,
        score_factors=None,
    ):
        num_classes = scores.shape[1]
        bboxes = np.broadcast_to(
            bboxes[:, None],
            (bboxes.shape[0], num_classes, 4),
        )
        valid_mask = scores > score_th
        bboxes = bboxes[valid_mask]

        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]

        np_labels = valid_mask.nonzero()[1]

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            score_th,
            nms_th,
        )

        if max_num > 0:
            indices = indices[:max_num]

        if len(indices) > 0:
            bboxes = bboxes[indices]
            scores = scores[indices]
            np_labels = np_labels[indices]
            return bboxes, scores, np_labels
        else:
            return np.array([]), np.array([]), np.array([])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='damoyolo_tinynasL20_T_192x320.onnx',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.4,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.85,
        help='NMS IoU threshold',
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    score_th = args.score_th
    nms_th = args.nms_th

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    model = DAMOYOLO(
        model_path,
        providers=[
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # Capture
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Execute inference
        bboxes, scores, class_ids = model(frame, nms_th=nms_th)

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            bboxes,
            scores,
            class_ids,
        )

        cv2.imshow('DAMO-YOLO ONNX Sample', debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    bboxes,
    scores,
    class_ids,
):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        # Bounding Box
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # Class ID, Score
        score = '%.2f' % score
        text = '%s:%s' % (str(class_id), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    # Elapsed time
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
