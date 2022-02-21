import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    # Pre process:Resize, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = input_image.astype('float32')
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    result = interpreter.get_tensor(output_details[0]['index'])

    # Post process:convert numpy array
    result = np.array(result[0])

    return result


def cos_similarity(X, Y):
    Y = Y.T

    # (128,) x (n, 128) = (n,)
    result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=
        'face_recognition_sface_2021dec_112x112/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='112,112',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Read image
    image01 = cv.imread('image01.jpg')
    image02 = cv.imread('image02.jpg')

    # Inference
    feature_vector01 = run_inference(interpreter, input_size, image01)
    feature_vector02 = run_inference(interpreter, input_size, image02)

    print(cos_similarity(feature_vector01, feature_vector02))


if __name__ == '__main__':
    main()