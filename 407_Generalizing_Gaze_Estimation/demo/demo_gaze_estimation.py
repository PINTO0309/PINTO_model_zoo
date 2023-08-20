import cv2
import argparse
import copy
import psutil
import onnxruntime
import numpy as np
import menpo.io as mio
from skimage import transform as trans
from typing import Optional, List, Tuple


class RetinaFaceONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx',
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
        """RetinaFaceONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path

        providers: Optional[List]
            Name of onnx execution providers
        """
        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        session_option.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
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
        self.mean = np.asarray([104, 117, 123], dtype=np.float32)

    def __call__(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        batchno_classid_score_x1y1x2y2_landms: np.ndarray
            [N, [batchno, classid, score, x1, y1, x2, y2, landms0, ..., landms9]]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self.__preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        batchno_classid_score_x1y1x2y2_landms = \
            self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )[0]

        return batchno_classid_score_x1y1x2y2_landms

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
        resized_image = resized_image[..., ::-1]
        resized_image = (resized_image - self.mean)
        resized_image = resized_image.transpose(swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image

def angles_from_vec(vec):
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z) - np.pi/2
    theta_x, theta_y = phi, theta
    return theta_x, theta_y

def vec_from_eye(eye, iris_lms_idx):
    p_iris = eye[iris_lms_idx] - eye[:32].mean(axis=0)
    vec = p_iris.mean(axis=0)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def angles_and_vec_from_eye(eye, iris_lms_idx):

    vec = vec_from_eye(eye, iris_lms_idx)
    theta_x, theta_y = angles_from_vec(vec)
    return theta_x, theta_y, vec

def vec_from_angles(rx, ry):
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    x1 = np.sin(np.pi/2 + rx) * np.cos(ry)
    y1 = np.sin(np.pi/2 + rx) * np.sin(ry)
    z1 = np.cos(np.pi/2 + rx)
    x, y, z = -z1, y1, -x1
    vec = np.array([x, y, z])
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(
        translation=(-1 * cx, -1 * cy)
    )
    t3 = trans.SimilarityTransform(
        rotation=rot
    )
    t4 = trans.SimilarityTransform(
        translation=(
            output_size / 2,
            output_size / 2
        )
    )
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(
        data,
        M,
        (output_size, output_size),
        borderValue=0.0
    )
    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]
    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
    return new_pts

def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class GazeHandler():
    def __init__(
        self,
        detector,
        model_path='generalizing_gaze_estimation_with_weak_supervision_from_synthetic_views_Nx3x160x160.onnx',
        res_eyes_path='assets/eyes3d.pkl',
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
        enable_3d_rendering = False,
    ):
        self.detector = detector

        eyes_mean = mio.import_pickle(res_eyes_path)
        idxs481 = eyes_mean['mask481']['idxs']
        self.tri481 = eyes_mean['mask481']['trilist']
        self.iris_idx_481 = eyes_mean['mask481']['idxs_iris']

        self.mean_l = eyes_mean['left_points'][idxs481][:, [0, 2, 1]]
        self.mean_r = eyes_mean['right_points'][idxs481][:, [0, 2, 1]]

        self.num_face = 1103
        self.num_eye = 481
        self.input_size = 160
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        session_option.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]
        self.enable_3d_rendering = enable_3d_rendering

    def draw_item(self, eimg, item):
        #bbox, kps, eye_kps = item
        eye_kps = item
        eye_l = eye_kps[:self.num_eye,:]
        eye_r = eye_kps[self.num_eye:,:]
        for _eye in [eye_l, eye_r]:
            tmp = _eye[:,0].copy()
            _eye[:,0] = _eye[:,1].copy()
            _eye[:,1] = tmp

        if self.enable_3d_rendering:
            for _eye in [eye_l, eye_r]:
                _kps = _eye[self.iris_idx_481,:].astype(np.int32)
                for l in range(_kps.shape[0]):
                    color = (0, 255, 0)
                    cv2.circle(eimg, (_kps[l][1], _kps[l][0]), 4, color, 4)
                for _tri in self.tri481:
                    color = (0, 0, 255)
                    for k in range(3):
                        ix = _tri[k]
                        iy = _tri[(k+1)%3]
                        x = _eye[ix,:2].astype(np.int32)[::-1]
                        y = _eye[iy,:2].astype(np.int32)[::-1]
                        cv2.line(eimg, x, y, color, 1)

        theta_x_l, theta_y_l, vec_l = angles_and_vec_from_eye(eye_l, self.iris_idx_481)
        theta_x_r, theta_y_r, vec_r = angles_and_vec_from_eye(eye_r, self.iris_idx_481)
        gaze_pred = np.array([(theta_x_l + theta_x_r) / 2, (theta_y_l + theta_y_r) / 2])

        diag = np.sqrt(float(eimg.shape[0]*eimg.shape[1]))

        eye_pos_left = eye_l[self.iris_idx_481].mean(axis=0)[[0, 1]]
        eye_pos_right = eye_r[self.iris_idx_481].mean(axis=0)[[0, 1]]

        ## pred
        gaze_pred = np.array([theta_x_l, theta_y_l])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_left[1], eye_pos_left[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        color = (0,255,0)
        cv2.line(eimg, x, y, color, 5)

        gaze_pred = np.array([theta_x_r, theta_y_r])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_right[1], eye_pos_right[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        color = (0,255,0)
        cv2.line(eimg, x, y, color, 5)
        return eimg

    def draw_on(self, eimg, results):
        face_sizes = [ (x[0][2] - x[0][0]) for x in results]
        max_index = np.argmax(face_sizes)
        max_face_size = face_sizes[max_index]
        rescale = 300.0 / max_face_size
        oimg = eimg.copy()
        eimg = cv2.resize(eimg, None, fx=rescale, fy=rescale)
        for pred in results:
            _, _, eye_kps = pred
            eye_kps = eye_kps.copy()
            eye_kps *= rescale
            eimg = self.draw_item(eimg, eye_kps)
        eimg = cv2.resize(eimg, (oimg.shape[1], oimg.shape[0]))
        return eimg

    def get(self, img):
        results = []
        batchno_classid_score_x1y1x2y2_landms = self.detector(img)
        if len(batchno_classid_score_x1y1x2y2_landms)==0:
            return results
        image_width = img.shape[1]
        image_height = img.shape[0]
        face_imgs = []
        face_kps = []
        Ms = []
        for face in batchno_classid_score_x1y1x2y2_landms:
            x_min = max(int(face[3]), 0)
            y_min = max(int(face[4]), 0)
            x_max = min(int(face[5]), image_width)
            y_max = min(int(face[6]), image_height)

            bbox = [x_min, y_min, x_max, y_max]
            kps = face[7:]
            kps_right_eye = np.asarray([int(face[7]), int(face[8])], dtype=np.int32) # [x, y]
            kps_left_eye = np.asarray([int(face[9]), int(face[10])], dtype=np.int32) # [x, y]
            width = x_max - x_min
            center = (kps_left_eye + kps_right_eye) / 2.0 # (lx + rx) / 2, (ly + ry) / 2

            _size = max(width/1.5, np.abs(kps_right_eye[0] - kps_left_eye[0]) ) * 1.5
            rotate = 0
            _scale = self.input_size  / _size
            aimg, M = transform(img, center, self.input_size, _scale, rotate)
            aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)

            face_imgs.append(aimg)
            face_kps.append(kps)
            Ms.append(M)

        input_face_images = np.asarray(face_imgs, dtype=np.float32)
        input_face_images = input_face_images.transpose([0,3,1,2])
        input_face_images = (input_face_images / 255.0 - 0.5) / 0.5

        opreds = \
            self.onnx_session.run(
                self.output_names,
                {input_name: input_face_images for input_name in self.input_names},
            )[0]

        for opred, face_kp, M in zip(opreds, face_kps, Ms):
            IM = cv2.invertAffineTransform(M)
            pred = trans_points(opred, IM)
            result = (bbox, face_kp, pred)
            results.append(result)

        return results


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
        '-dm',
        '--detector_model',
        type=str,
        default='retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx',
    )
    parser.add_argument(
        '-pm',
        '--predictor_model',
        type=str,
        default='generalizing_gaze_estimation_with_weak_supervision_from_synthetic_views_Nx3x160x160.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cuda',
        choices=['cpu','cuda','tensorrt'],
    )
    parser.add_argument(
        '-etr',
        '--enable_3d_rendering',
        action='store_true',
    )
    args = parser.parse_args()

    cap_device: int = args.device
    if args.movie is not None:
        cap_device = args.movie

    providers = None
    if args.provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.provider == 'tensorrt':
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
    enable_3d_rendering: bool = args.enable_3d_rendering

    cap = cv2.VideoCapture(cap_device)
    cap_width = 640
    cap_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    detector = RetinaFaceONNX(
        model_path=args.detector_model,
        providers=providers,
    )
    handler = GazeHandler(
        detector=detector,
        model_path=args.predictor_model,
        providers=providers,
        enable_3d_rendering=enable_3d_rendering,
    )
    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)
        debug_image = cv2.resize(debug_image, (640,480))
        results = handler.get(debug_image)
        if len(results) > 0:
            debug_image = handler.draw_on(debug_image, results)
        video_writer.write(debug_image)
        cv2.imshow('Generalizing Gaze Estimation', debug_image)
        key = cv2.waitKey(1) \
            if args.movie is None or args.movie[-4:] == '.mp4' else cv2.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

