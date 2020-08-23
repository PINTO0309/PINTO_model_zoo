### tensorflow==2.3.0-rc0

import tensorflow as tf
import numpy as np

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        # image = image - 127.5
        # image = image * 0.007843
        yield [image]

raw_test_data = np.load('selfie2anime_dataset.npy', allow_pickle=True)

# Full Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('selfie2anime_256x256_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - selfie2anime_256x256_full_integer_quant.tflite")