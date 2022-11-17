#!/usr/bin/env python

import copy

import cv2
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from typing import Tuple, Optional, List


class YOLOv5ONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'yolov5l6_ball_1088x1920_130050_post.onnx',
        class_score_th: Optional[float] = 0.30,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """YOLOv5ONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for YOLOv5

        class_score_th: Optional[float]

        class_score_th: Optional[float]
            Score threshold. Default: 0.30

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
        """YOLOv5ONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        face_boxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]

        face_scores: np.ndarray
            Predicted face box scores: [facecount, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        scores, boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )

        # PostProcess
        face_boxes, face_scores = self.__postprocess(
            image=temp_image,
            scores=scores,
            boxes=boxes,
        )

        return face_boxes, face_scores


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
        scores: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        scores: np.ndarray
            float32[N, 1]

        boxes: np.ndarray
            int64[N, 6]

        Returns
        -------
        faceboxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]

        facescores: np.ndarray
            Predicted face box confs: [facecount, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Ball Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0
            classid -> always 0: "Ball"

        scores: float32[N,1],
        batchno_classid_y1x1y2x2: int64[N,6],
        """
        scores = scores
        keep_idxs = scores[:, 0] > self.class_score_th
        scores_keep = scores[keep_idxs, :]
        boxes_keep = boxes[keep_idxs, :]
        faceboxes = []
        facescores = []

        if len(boxes_keep) > 0:
            for box, score in zip(boxes_keep, scores_keep):
                x_min = max(int(box[3]), 0)
                y_min = max(int(box[2]), 0)
                x_max = min(int(box[5]), image_width)
                y_max = min(int(box[4]), image_height)

                faceboxes.append(
                    [x_min, y_min, x_max, y_max]
                )
                facescores.append(
                    score
                )

        return np.asarray(faceboxes), np.asarray(facescores)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='yolov5l6_ball_1088x1920_130050_post.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default='demo.mp4',
    )
    args = parser.parse_args()

    model = YOLOv5ONNX(
        model_path=args.model,
    )

    cap = cv2.VideoCapture(args.video)
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
        face_boxes, face_scores = model(debug_image)
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        cv2.putText(
            debug_image,
            f'{fps:.1f} FPS (inferece only)',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'{fps:.1f} FPS (inferece only)',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        for face_box, face_score in zip(face_boxes, face_scores):
            cv2.rectangle(
                debug_image,
                (face_box[0], face_box[1]),
                (face_box[2], face_box[3]),
                (255,255,255),
                2,
            )
            cv2.rectangle(
                debug_image,
                (face_box[0], face_box[1]),
                (face_box[2], face_box[3]),
                (0,255,0),
                1,
            )
            cv2.putText(
                debug_image,
                f'{face_score[0]:.2f}',
                (
                    face_box[0],
                    face_box[1]-10 if face_box[1]-10 > 0 else 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{face_score[0]:.2f}',
                (
                    face_box[0],
                    face_box[1]-10 if face_box[1]-10 > 0 else 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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
