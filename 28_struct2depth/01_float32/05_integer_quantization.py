### tf-nightly-2.2.0-dev20200502
import tensorflow as tf
import numpy as np

def representative_dataset_gen_kitti():
  for image in raw_test_data_hand:
    image = tf.image.resize(image.astype(np.float32), (128, 416))
    image = image[np.newaxis,:,:,:]
    image = image / 255
    yield [image]

def representative_dataset_gen_cityscapes():
  for image in raw_test_data_hand:
    image = tf.image.resize(image.astype(np.float32), (128, 416))
    image = image[np.newaxis,:,:,:]
    image = image / 255
    yield [image]

raw_test_data_hand = np.load('calibration_data_img_kitti.npy', allow_pickle=True)
raw_test_data_joint = np.load('calibration_data_img_cityscapes.npy', allow_pickle=True)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_kitti')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen_kitti
tflite_quant_model = converter.convert()
with open('struct2depth_128x416_kitti_depth_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - struct2depth_128x416_kitti_depth_integer_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_cityscapes')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen_cityscapes
tflite_quant_model = converter.convert()
with open('struct2depth_128x416_cityscapes_depth_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - struct2depth_128x416_cityscapes_depth_integer_quant.tflite")