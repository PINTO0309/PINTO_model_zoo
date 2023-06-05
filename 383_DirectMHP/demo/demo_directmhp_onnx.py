

import sys
import cv2
import time
import math
import copy
import psutil
import argparse
import onnxruntime
import numpy as np
from math import cos, sin
from typing import Tuple, Optional, List, Iterable, NamedTuple


class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


class Box(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    yaw: float
    pitch: float
    roll: float
    score: float


class DirectMHPONNX(object):
    def __init__(
        self,
        model_file_path: Optional[str] = 'directmhp_cmu_m_post_512x640.onnx',
        class_score_th: Optional[float] = 0.20,
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
        """DirectMHPONNX

        Parameters
        ----------
        model_file_path: Optional[str]
            ONNX file path for DirectMHP

        class_score_th: Optional[float]
            Score threshold. Default: 0.20

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
        session_option.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.onnx_session = onnxruntime.InferenceSession(
            model_file_path,
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
    ) -> Iterable[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        face_boxes: Iterable[Box]
            Predicted face boxes
            [N, [x_min, y_min, x_max, y_max, yaw, pitch, roll, score]]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        batchno_classid_x1y1x2y2_score_pitchyawroll = \
            self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )[0]

        # PostProcess
        # face_boxes: [N, [x_min, y_min, x_max, y_max, yaw, pitch, roll, score]]
        face_boxes: Iterable[Box] = \
            self.__postprocess(
                image=temp_image,
                batchno_classid_x1y1x2y2_score_pitchyawroll=batchno_classid_x1y1x2y2_score_pitchyawroll,
            )

        return face_boxes


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
        batchno_classid_x1y1x2y2_score_pitchyawroll: np.ndarray,
    ) -> Iterable[Box]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        batchno_classid_x1y1x2y2_score_pitchyawroll: np.ndarray
            float32[N, 10]
            [N, [batchno, classid, x1, y1, x2, y2, score, pitch, yaw, roll]]

        Returns
        -------
        faceboxes: Iterable[Box]
            [N, [x_min, y_min, x_max, y_max, yaw, pitch, roll, score]]
        """
        image_height: int = image.shape[0]
        image_width: int = image.shape[1]
        input_height: int = self.input_shapes[0][2]
        input_width: int = self.input_shapes[0][3]
        # Head Detector is
        #     N -> Number of boxes detected
        #     batchno -> always 0: BatchNo.0
        #     classid -> always 0: "Head"
        #
        # faceboxes: float32[N, 8]
        #     [N, [x_min, y_min, x_max, y_max, yaw, pitch, roll, score]]
        keep_idxs = batchno_classid_x1y1x2y2_score_pitchyawroll[:, 6] > self.class_score_th

        if batchno_classid_x1y1x2y2_score_pitchyawroll.size > 0:
            batchno_classid_x1y1x2y2_score_pitchyawroll_keep = \
                batchno_classid_x1y1x2y2_score_pitchyawroll[keep_idxs, :]

            if len(batchno_classid_x1y1x2y2_score_pitchyawroll_keep) > 0:
                for box in batchno_classid_x1y1x2y2_score_pitchyawroll_keep:
                    scale_ratio_width: float = float(image_width / input_width)
                    scale_ratio_height: float = float(image_height / input_height)
                    cx: float = (box[4] + box[2]) / 2.0 * scale_ratio_width
                    cy: float = (box[5] + box[3]) / 2.0 * scale_ratio_height
                    real_width: float = (box[4] - box[2]) * scale_ratio_width
                    real_height: float = (box[5] - box[3]) * scale_ratio_height
                    x_min: float = max(cx - real_width / 2.0, 0.0)
                    y_min: float = max(cy - real_height / 2.0, 0.0)
                    x_max: float = min(cx + real_width / 2.0, image_width)
                    y_max: float = min(cy + real_height / 2.0, image_height)
                    score: float = float(box[6])
                    pitch: float = float(box[7])
                    yaw: float = float(box[8])
                    roll: float = float(box[9])
                    yield \
                        Box(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            yaw=yaw,
                            pitch=pitch,
                            roll=roll,
                            score=score,
                        )

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    if math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll):
        return img
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def main(args):
    model_file_path = args.model_file_path
    # DirectMHP
    directmhp_head = \
        DirectMHPONNX(
            model_file_path=model_file_path,
            class_score_th=0.40,
        )
    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)
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

    print('')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # ============================================================= DirectMHP
        # face_boxes: [N, [x_min, y_min, x_max, y_max, yaw, pitch, roll, score]]
        face_boxes: Iterable[Box] = directmhp_head(frame)
        canvas = copy.deepcopy(frame)

        for face_box in face_boxes:
            sys.stdout.write(
                f'  ' +
                f'{Color.GREEN}yaw:{Color.RESET} {face_box.yaw:.2f}, ' +
                f'{Color.GREEN}pitch:{Color.RESET} {face_box.pitch:.2f}, ' +
                f'{Color.GREEN}roll:{Color.RESET} {face_box.roll:.2f}' +
                f'\r'
            )
            sys.stdout.flush()
            # BBox draw
            deg_norm: float = 1.0 - abs(face_box.yaw / 180)
            blue: int = int(255 * deg_norm)
            cv2.rectangle(
                canvas,
                (int(face_box.x_min), int(face_box.y_min)),
                (int(face_box.x_max), int(face_box.y_max)),
                color=(blue, 0, 255-blue),
                thickness=2
            )
            # Axis Draw
            draw_axis(
                canvas,
                face_box.yaw,
                face_box.pitch,
                face_box.roll,
                tdx=(face_box.x_min+face_box.x_max)/2,
                tdy=(face_box.y_min+face_box.y_max)/2,
                size=abs(face_box.x_max-face_box.x_min)//3
            )
            cv2.putText(
                canvas,
                f'yaw: {np.round(face_box.yaw)}',
                (int(face_box.x_min), int(face_box.y_min)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 255, 0),
                1
            )
            cv2.putText(
                canvas,
                f'pitch: {np.round(face_box.pitch)}',
                (int(face_box.x_min), int(face_box.y_min) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 255, 0),
                1
            )
            cv2.putText(
                canvas,
                f'roll: {np.round(face_box.roll)}',
                (int(face_box.x_min), int(face_box.y_min)-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 255, 0),
                1
            )

        time_txt = f'{(time.time()-start)*1000:.2f} ms (inference+post-process)'
        cv2.putText(
            canvas,
            time_txt,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            time_txt,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print('')
            break

        cv2.imshow(WINDOWS_NAME, canvas)
        video_writer.write(canvas)

    cv2.destroyAllWindows()

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        '--height_width',
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    parser.add_argument(
        '--model_file_path',
        type=str,
        default='directmhp_cmu_m_post_512x640.onnx',
    )
    args = parser.parse_args()
    main(args)
