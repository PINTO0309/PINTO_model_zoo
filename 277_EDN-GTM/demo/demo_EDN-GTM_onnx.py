#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import time
import math
import argparse
import cv2 as cv
import numpy as np
import onnxruntime

def DarkChannel(im,sz):
    b, g, r = cv.split(im)
    dc = cv.min(cv.min(r, g), b)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (sz, sz))
    dark = cv.erode(dc, kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]
    atmsum = np.zeros([1, 3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission

def TransmissionRefine(im,et):
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t

def estimate_transmission(src):
    I = src.astype(np.float64) / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    return t

def Guidedfilter(im,p,r,eps):
    mean_I = cv.boxFilter(im,cv.CV_64F,(r, r))
    mean_p = cv.boxFilter(p, cv.CV_64F,(r, r))
    mean_Ip = cv.boxFilter(im * p,cv.CV_64F,(r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv.boxFilter(im * im,cv.CV_64F,(r, r))
    var_I   = mean_II - mean_I * mean_I
    a = cov_Ip/(var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv.boxFilter(a,cv.CV_64F,(r, r))
    mean_b = cv.boxFilter(b,cv.CV_64F,(r, r))
    q = mean_a * im + mean_b
    return q

def preprocess_depth_img(cv_img, input_size):
    cv_img = cv.resize(cv_img, (input_size[1], input_size[0]))
    img = np.array(cv_img)
    img = np.reshape(img, (input_size[0], input_size[1], 1))
    img = 2 * (img - 0.5)
    return img

def preprocess_cv2_image(cv_img, input_size):
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    cv_img = cv.resize(cv_img, (input_size[1], input_size[0]))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def run_inference(onnx_session, input_size, image):
    # Pre process
    image1 = copy.deepcopy(image)
    image2 = copy.deepcopy(image)

    t = estimate_transmission(image1)
    t = preprocess_depth_img(t, input_size)
    t = t.astype(np.float32)

    image2 = preprocess_cv2_image(image2, input_size)
    image2 = image2.astype(np.float32)

    input_image = np.concatenate((image2, t), axis=2)
    input_image = np.reshape(input_image, (1, input_size[0], input_size[1], 4))

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})
    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    output_image = np.squeeze(result[0])
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='ihaze_generator_384x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='384,640',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter('output.mp4', fourcc, cap_fps, (w,h))
    window_name = 'EDN-GTM test'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.dirname(model_path),
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider'
        ],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        output_image = cv.resize(
            output_image,
            dsize=(frame_width, frame_height)
        )
        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            output_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        combined_img = np.vstack([debug_image, output_image])
        cv.imshow(window_name, combined_img)
        out.write(combined_img)

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()