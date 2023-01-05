import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, image):
    # TFLite Infomation
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape']
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process:Resize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Post process:convert numpy array
    output_details = interpreter.get_output_details()
    bottom_color_point = interpreter.get_tensor(output_details[0]['index'])
    top_color_point = interpreter.get_tensor(output_details[1]['index'])
    attribute = interpreter.get_tensor(output_details[2]['index'])

    attribute = np.squeeze(attribute)
    top_color_point = np.squeeze(top_color_point)
    bottom_color_point = np.squeeze(bottom_color_point)

    return attribute, top_color_point, bottom_color_point


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.tflite',
    )

    args = parser.parse_args()
    model_path = args.model

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Read image
    image = cv.imread('sample.jpg')

    # Inference
    attribute, top_point, bottom_point = run_inference(interpreter, image)

    print('is_male        :', attribute[0])
    print('has_bag        :', attribute[1])
    print('has_backpack   :', attribute[2])
    print('has_hat        :', attribute[3])
    print('has_longsleeves:', attribute[4])
    print('has_longpants  :', attribute[5])
    print('has_longhair   :', attribute[6])
    print('has_coat_jacket:', attribute[7])

    image_width, image_height = image.shape[1], image.shape[0]

    top_point_x = int(top_point[0] * image_width)
    top_point_y = int(top_point[1] * image_height)
    cv.circle(image, (top_point_x, top_point_y), 5, (0, 0, 255), -1)

    bottom_point_x = int(bottom_point[0] * image_width)
    bottom_point_y = int(bottom_point[1] * image_height)
    cv.circle(image, (bottom_point_x, bottom_point_y), 5, (255, 0, 0), -1)

    cv.imshow('person-attributes-recognition-crossroad-0230', image)
    cv.waitKey(-1)


if __name__ == '__main__':
    main()
