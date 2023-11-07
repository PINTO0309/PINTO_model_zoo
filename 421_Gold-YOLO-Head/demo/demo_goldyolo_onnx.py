#!/usr/bin/env python

import copy
import cv2
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from typing import Tuple, Optional, List


class GoldYOLOONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'gold_yolo_n_hand_post_0333_0.4040_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """GoldYOLOONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for GoldYOLO

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
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
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv7ONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )[0]

        # PostProcess
        result_boxes, result_scores = \
            self.__postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes, result_scores


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

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
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_y1x1y2x2_score: float32[N,7]
        """
        result_boxes = []
        result_scores = []
        if len(boxes) > 0:
            scores = boxes[:, 6:7]
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(box[2], 0) * image_width / self.input_shapes[0][3])
                    y_min = int(max(box[3], 0) * image_height / self.input_shapes[0][2])
                    x_max = int(min(box[4], self.input_shapes[0][3]) * image_width / self.input_shapes[0][3])
                    y_max = int(min(box[5], self.input_shapes[0][2]) * image_height / self.input_shapes[0][2])

                    result_boxes.append(
                        [x_min, y_min, x_max, y_max]
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
        default='gold_yolo_n_hand_post_0333_0.4040_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    args = parser.parse_args()

    model = GoldYOLOONNX(
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

        start_time = time.time()
        boxes, scores = model(debug_image)
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        cv2.putText(
            debug_image,
            f'{fps:.1f} FPS (inferece + post-process)',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'{fps:.1f} FPS (inferece + post-process)',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        for box, score in zip(boxes, scores):
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
                (0,0,255),
                1,
            )
            cv2.putText(
                debug_image,
                f'{score[0]:.2f}',
                (
                    box[0],
                    box[1]-10 if box[1]-10 > 0 else 10
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
                    box[0],
                    box[1]-10 if box[1]-10 > 0 else 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
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
