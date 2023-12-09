#!/usr/bin/env python

import copy
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import SignatureRunner
from argparse import ArgumentParser
from typing import Tuple, Optional


class YOLOXTFLite(object):
    def __init__(
        self,
        model_path: Optional[str] = 'yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite',
        class_score_th: Optional[float] = 0.35,
    ):
        """YOLOXTFLite

        Parameters
        ----------
        model_path: Optional[str]
            TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_shapes = [
            input['shape'] for input in input_details
        ]
        self.input_names = [
            input['name'] for input in input_details
        ]
        self.input_indexes = [
            input['index'] for input in input_details
        ]

        self.output_shapes = [
            output['shape'] for output in output_details
        ]
        self.output_names = [
            output['name'] for output in output_details
        ]
        self.output_indexes = [
            output['index'] for output in output_details
        ]
        self.model: SignatureRunner = self.interpreter.get_signature_runner()

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOXTFLite

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        outputs = self.model(**{self.input_names[0]: inferece_image})
        boxes = outputs[self.output_names[0]]

        # PostProcess
        result_boxes, result_scores = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes, result_scores


    def _preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][2]),
                int(self.input_shapes[0][1]),
            )
        )
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )

        return resized_image


    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ):
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes = []
        result_scores = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    class_id = int(box[1])
                    x_min = int(max(0, box[3]) * image_width / self.input_shapes[0][2])
                    y_min = int(max(0, box[4]) * image_height / self.input_shapes[0][1])
                    x_max = int(min(box[5], self.input_shapes[0][2]) * image_width / self.input_shapes[0][2])
                    y_max = int(min(box[6], self.input_shapes[0][1]) * image_height / self.input_shapes[0][1])
                    result_boxes.append(
                        [x_min, y_min, x_max, y_max, class_id]
                    )
                    result_scores.append(
                        score
                    )

        return np.asarray(result_boxes), np.asarray(result_scores)


def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    args = parser.parse_args()

    model = YOLOXTFLite(
        model_path=args.model,
    )

    cap = cv2.VideoCapture(
        int(args.video) if is_parsable_to_int(args.video) else args.video
    )
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h),
    )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)

        start_time = time.perf_counter()
        boxes, scores = model(debug_image)
        elapsed_time = time.perf_counter() - start_time
        cv2.putText(
            debug_image,
            f'{elapsed_time*1000:.2f} ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'{elapsed_time*1000:.2f} ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        for box, score in zip(boxes, scores):
            classid: int = box[4]
            color = (255,255,255)
            if classid == 0:
                color = (255,0,0)
            elif classid == 1:
                color = (0,0,255)
            elif classid == 2:
                color = (0,255,0)
            cv2.rectangle(
                debug_image,
                (box[0], box[1]),
                (box[2], box[3]),
                (255,255,255),
                2,
            )
            cv2.rectangle(
                debug_image,
                (box[0], box[1]),
                (box[2], box[3]),
                color,
                1,
            )
            cv2.putText(
                debug_image,
                f'{score[0]:.2f}',
                (
                    box[0] if box[0]+50 < debug_image.shape[1] else debug_image.shape[1]-50,
                    box[1]-10 if box[1]-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{score[0]:.2f}',
                (
                    box[0] if box[0]+50 < debug_image.shape[1] else debug_image.shape[1]-50,
                    box[1]-10 if box[1]-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", debug_image)
        video_writer.write(debug_image)

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()

if __name__ == "__main__":
    main()
