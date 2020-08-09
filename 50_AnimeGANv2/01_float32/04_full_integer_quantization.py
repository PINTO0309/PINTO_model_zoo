### tensorflow==2.3.0

import tensorflow as tf
import numpy as np

def representative_dataset_gen():
  for count, image in enumerate(raw_test_data):
    print('image.shape:', count, image.shape)
    image = tf.image.resize(image, (256, 256))
    image = image[np.newaxis,:,:,:]
    image = image / 127.5 - 1.0
    yield [image]

# Full Integer Quantization - Input/Output=float32
raw_test_data = np.load('animeganv2_dataset_hayao_256x256.npy', allow_pickle=True)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_Hayao')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('animeganv2_hayao_256x256_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - animeganv2_hayao_256x256_full_integer_quant.tflite")

# Full Integer Quantization - Input/Output=float32
raw_test_data = np.load('animeganv2_dataset_paprika_256x256.npy', allow_pickle=True)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_Paprika')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('animeganv2_paprika_256x256_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - animeganv2_paprika_256x256_full_integer_quant.tflite")
