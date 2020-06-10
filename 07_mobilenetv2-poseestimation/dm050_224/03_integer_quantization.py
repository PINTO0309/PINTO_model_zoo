### tensorflow==2.2.0

import tensorflow as tf
import numpy as np

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        yield [image]

raw_test_data = np.load('calibration_data_img_person.npy', allow_pickle=True)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('mobilenet_v2_pose_256_256_dm050_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - mobilenet_v2_pose_256_256_dm050_integer_quant.tflite")