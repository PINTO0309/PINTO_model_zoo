#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time

import cv2
import numpy as np

import tensorflow as tf

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


class NanoDetTFLite(object):
    # Constant definition for post process
    STRIDES = (8, 16, 32)
    REG_MAX = 7
    PROJECT = np.arange(REG_MAX + 1)

    # Constant definition for Standardization
    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    MEAN = MEAN.reshape(1, 1, 3)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    STD = STD.reshape(1, 1, 3)

    def __init__(
        self,
        model_path='model_float16_quant.tflite',
        input_shape=320,
        class_score_th=0.35,
        nms_th=0.6,
        num_threads=1,
    ):
        self.input_shape = (input_shape, input_shape)

        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # load model
        self.interpreter = Interpreter(model_path=model_path,
                                       num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Calculate grid points for each stride
        self.grid_points = []
        for index in range(len(self.STRIDES)):
            grid_point = self._make_grid_point(
                (int(self.input_shape[0] / self.STRIDES[index]),
                 int(self.input_shape[1] / self.STRIDES[index])),
                self.STRIDES[index],
            )
            self.grid_points.append(grid_point)

    def inference(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # Pre process: Standardization, Reshape
        resize_image, new_height, new_width, top, left = self._resize_image(
            temp_image)
        x = self._pre_process(resize_image)

        # Inference execution
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()

        results = []
        results.append(
            self.interpreter.get_tensor(
                self.output_details[5]['index']))  # cls_pred_stride_8
        results.append(
            self.interpreter.get_tensor(
                self.output_details[4]['index']))  # dis_pred_stride_8
        results.append(
            self.interpreter.get_tensor(
                self.output_details[3]['index']))  # cls_pred_stride_16
        results.append(
            self.interpreter.get_tensor(
                self.output_details[2]['index']))  # dis_pred_stride_16
        results.append(
            self.interpreter.get_tensor(
                self.output_details[1]['index']))  # cls_pred_stride_32
        results.append(
            self.interpreter.get_tensor(
                self.output_details[0]['index']))  # dis_pred_stride_32

        # Post-process: NMS, grid -> coordinate transformation
        bboxes, scores, class_ids = self._post_process(results)

        # Post-process: Convert coordinates to fit image size
        ratio_height = image_height / new_height
        ratio_width = image_width / new_width
        for i in range(bboxes.shape[0]):
            bboxes[i, 0] = max(int((bboxes[i, 0] - left) * ratio_width), 0)
            bboxes[i, 1] = max(int((bboxes[i, 1] - top) * ratio_height), 0)
            bboxes[i, 2] = min(
                int((bboxes[i, 2] - left) * ratio_width),
                image_width,
            )
            bboxes[i, 3] = min(
                int((bboxes[i, 3] - top) * ratio_height),
                image_height,
            )
        return bboxes, scores, class_ids

    def _make_grid_point(self, grid_size, stride):
        grid_height, grid_width = grid_size

        shift_x = np.arange(0, grid_width) * stride
        shift_y = np.arange(0, grid_height) * stride

        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()

        cx = xv + 0.5 * (stride - 1)
        cy = yv + 0.5 * (stride - 1)

        return np.stack((cx, cy), axis=-1)

    def _resize_image(self, image, keep_ratio=True):
        top, left = 0, 0
        new_height, new_width = self.input_shape[0], self.input_shape[1]

        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                new_height = self.input_shape[0]
                new_width = int(self.input_shape[1] / hw_scale)

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                left = int((self.input_shape[1] - new_width) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    0,
                    0,
                    left,
                    self.input_shape[1] - new_width - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                new_height = int(self.input_shape[0] * hw_scale)
                new_width = self.input_shape[1]

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                top = int((self.input_shape[0] - new_height) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    top,
                    self.input_shape[0] - new_height - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            resize_image = cv2.resize(
                image,
                self.input_shape,
                interpolation=cv2.INTER_AREA,
            )

        return resize_image, new_height, new_width, top, left

    def _pre_process(self, image):
        # Standardization
        image = image.astype(np.float32)
        image = (image - self.MEAN) / self.STD

        # Reshape
        image = image.reshape(-1, self.input_shape[0], self.input_shape[1], 3)

        return image

    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _post_process(self, predict_results):
        class_scores = predict_results[::2]
        bbox_predicts = predict_results[1::2]

        bboxes, scores, class_ids = self._get_bboxes_single(
            class_scores,
            bbox_predicts,
            1,
            rescale=False,
        )

        return bboxes.astype(np.int32), scores, class_ids

    def _get_bboxes_single(
        self,
        class_scores,
        bbox_predicts,
        scale_factor,
        rescale=False,
        topk=1000,
    ):
        bboxes = []
        scores = []

        # Convert bounding box coordinates for each stride
        for stride, class_score, bbox_predict, grid_point in zip(
                self.STRIDES, class_scores, bbox_predicts, self.grid_points):
            # Dimension adjustment
            if class_score.ndim == 3:
                class_score = class_score.squeeze(axis=0)
            if bbox_predict.ndim == 3:
                bbox_predict = bbox_predict.squeeze(axis=0)

            # Convert the bounding box to relative coordinates and relative distance
            bbox_predict = bbox_predict.reshape(-1, self.REG_MAX + 1)
            bbox_predict = self._softmax(bbox_predict, axis=1)
            bbox_predict = np.dot(bbox_predict, self.PROJECT).reshape(-1, 4)
            bbox_predict *= stride

            # Target in descending order of score
            if 0 < topk < class_score.shape[0]:
                max_scores = class_score.max(axis=1)
                topk_indexes = max_scores.argsort()[::-1][0:topk]

                grid_point = grid_point[topk_indexes, :]
                bbox_predict = bbox_predict[topk_indexes, :]
                class_score = class_score[topk_indexes, :]

            # Convert the bounding box to absolute coordinates
            x1 = grid_point[:, 0] - bbox_predict[:, 0]
            y1 = grid_point[:, 1] - bbox_predict[:, 1]
            x2 = grid_point[:, 0] + bbox_predict[:, 2]
            y2 = grid_point[:, 1] + bbox_predict[:, 3]
            x1 = np.clip(x1, 0, self.input_shape[1])
            y1 = np.clip(y1, 0, self.input_shape[0])
            x2 = np.clip(x2, 0, self.input_shape[1])
            y2 = np.clip(y2, 0, self.input_shape[0])
            bbox = np.stack([x1, y1, x2, y2], axis=-1)

            bboxes.append(bbox)
            scores.append(class_score)

        # Scale adjustment
        bboxes = np.concatenate(bboxes, axis=0)
        if rescale:
            bboxes /= scale_factor
        scores = np.concatenate(scores, axis=0)

        # Non-Maximum Suppression
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        class_ids = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        indexes = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            scores.tolist(),
            self.class_score_th,
            self.nms_th,
        )

        # Check the number of cases after NMS processing
        if len(indexes) > 0:
            bboxes = bboxes[indexes[:, 0]]
            scores = scores[indexes[:, 0]]
            class_ids = class_ids[indexes[:, 0]]
        else:
            bboxes = np.array([])
            scores = np.array([])
            class_ids = np.array([])

        return bboxes, scores, class_ids


if __name__ == '__main__':
    # Initialize NanoDetTFlite Class
    nanodet = NanoDetTFLite(
        model_path='saved_model_nanodet_320x320/model_float16_quant.tflite',
        input_shape=320,
    )

    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        bboxes, scores, class_ids = nanodet.inference(frame)

        elapsed_time = time.time() - start_time

        # Draw
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Bounding Box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Class ID, Score
            score = '%.2f' % score
            text = '%s:%s' % (str(class_id), score)
            cv2.putText(debug_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Inference elapsed time
        text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
        text = text + 'ms'
        cv2.putText(debug_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('NanoDet TFLite Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()
