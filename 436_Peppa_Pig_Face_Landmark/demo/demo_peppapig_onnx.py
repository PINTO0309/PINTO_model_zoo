#!/usr/bin/env python

import copy
import cv2
import time
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _class_score_th: float = 0.35
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        class_score_th: Optional[float] = 0.35,
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
        self._runtime = runtime
        self._model_path = model_path
        self._class_score_th = class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        pass

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [N, classid, score, x1, y1, x2, y2]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                        )
                    )
        return result_boxes

class PeppaPig(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'peppapig_teacher_Nx3x256x256.onnx',
        providers: Optional[List] = None,
    ):
        """PeppaPig

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for FaceAlignment. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for FaceAlignment

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self._swap = (0,3,1,2)
        self._mean = np.asarray([0.0, 0.0, 0.0])
        self._std = np.asarray([1.0, 1.0, 1.0])


    def __call__(
        self,
        image: np.ndarray,
        face_boxes: List[Box],
    ) -> np.ndarray:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        face_boxes: List[Box]

        Returns
        -------
        result_landmarks: np.ndarray
            Predicted boxes: [N, 68, 2]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_images = \
            self._preprocess(
                image=temp_image,
                face_boxes=face_boxes,
            )

        result_landmarks: np.ndarray = np.asarray([], dtype=np.float32)

        if len(resized_images) > 0:
            # Inference
            outputs = super().__call__(input_datas=[resized_images])
            landmarks: np.ndarray = outputs[0]
            # PostProcess
            result_landmarks = \
                self._postprocess(
                    landmarks=landmarks,
                    face_boxes=face_boxes,
                )
        return result_landmarks

    def _preprocess(
        self,
        image: np.ndarray,
        face_boxes: List[Box],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        face_boxes: List[Box]
            Face boxes.

        Returns
        -------
        face_images_np: np.ndarray
            For inference, normalized face images.
        """
        image_height = image.shape[0]
        image_width = image.shape[1]
        face_images: List[np.ndarray] = []
        face_images_np: np.ndarray = np.asarray([], dtype=np.float32)

        if len(face_boxes) > 0:
            for face in face_boxes:
                cx = (face.x1 + face.x2) / 2
                cy = (face.y1 + face.y2) / 2
                w = abs(face.x2 - face.x1)
                h = abs(face.y2 - face.y1)
                face.x1 = max(0, int(cx - w * 0.5 * 1.3))
                face.y1 = max(0, int(cy - h * 0.5 * 1.3))
                face.x2 = min(int(cx + w * 0.5 * 1.3), image_width)
                face.y2 = min(int(cy + h * 0.5 * 1.3), image_height)
                face_image: np.ndarray = image[face.y1:face.y2, face.x1:face.x2, :]
                resized_face_image = \
                    cv2.resize(
                        face_image,
                        (
                            int(self._input_shapes[0][self._w_index]),
                            int(self._input_shapes[0][self._h_index]),
                        )
                    )
                face_images.append(resized_face_image)
            face_images_np = np.asarray(face_images, dtype=np.float32)
            face_images_np = face_images_np[..., ::-1]
            face_images_np = (face_images_np / 255.0 - self._mean) / self._std
            face_images_np = face_images_np.transpose(self._swap)
            face_images_np = face_images_np.astype(self._input_dtypes[0])
        return face_images_np

    def _postprocess(
        self,
        landmarks: np.ndarray,
        face_boxes: List[Box],
    ) -> np.ndarray:
        """_postprocess

        Parameters
        ----------
        landmarks: np.ndarray
            landmarks. [batch, 68, 2]

        face_boxes: List[Box]

        scaled_pad_and_scale_ratios: List[ScaledPad_and_ScaleRatio]

        Returns
        -------
        landmarks: np.ndarray
            Predicted landmarks: [batch, 68, 2]
        """
        if len(landmarks) > 0:
            for landmark, face_box in zip(landmarks, face_boxes):
                landmark[..., 0] = landmark[..., 0] * abs(face_box.x2 - face_box.x1)
                landmark[..., 1] = landmark[..., 1] * abs(face_box.y2 - face_box.y1)
                landmark[..., 0] = landmark[..., 0] + face_box.x1
                landmark[..., 1] = landmark[..., 1] + face_box.y1
        return landmarks.astype(np.float32)


def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-dm',
        '--detection_model',
        type=str,
        default='yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-fm',
        '--face_alignment_model',
        type=str,
        default='peppapig_teacher_Nx3x256x256.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='tensorrt',
    )
    args = parser.parse_args()

    providers: List[Tuple[str, Dict] | str] = None
    if args.execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'tensorrt':
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

    model_yolox = \
        YOLOX(
            model_path=args.detection_model,
            providers=providers,
        )
    model_facealign = \
        PeppaPig(
            model_path=args.face_alignment_model,
            providers=providers,
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
        boxes = model_yolox(debug_image)

        face_boxes: List[Box] = []
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)
            if classid == 0:
                color = (255,0,0)
            elif classid == 1:
                color = (0,0,255)
            elif classid == 2:
                color = (0,255,0)
            elif classid == 3:
                color = (0,200,255)

            # Make Face box
            if classid == 3:
                face_boxes.append(box)

            if classid != 3:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
            else:
                draw_dashed_rectangle(
                    image=debug_image,
                    top_left=(box.x1, box.y1),
                    bottom_right=(box.x2, box.y2),
                    color=color,
                    thickness=2,
                    dash_length=10
                )

        # Face alignment
        landmarks: np.ndarray = model_facealign(debug_image, face_boxes)

        elapsed_time = time.perf_counter() - start_time

        _ = [
            cv2.circle(debug_image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), 2) \
                for one_face_landmarks in landmarks \
                    for landmark in one_face_landmarks \
                        if landmark[2] > 0.75
        ]

        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 150, 196), 1, cv2.LINE_AA)

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
