### tensorflow==2.3.0
### https://github.com/mvoelk/ssd_detectors/blob/master/TBPP_evaluate.ipynb

import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.keras.backend as K


from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable
from ssd_data import InputGenerator
from tbpp_training import TBPPFocalLoss
from ssd_metric import fscore
from sl_metric import evaluate_polygonal_results
from utils.model import load_weights, calc_memory_usage
from utils.bboxes import rbox3_to_polygon, polygon_to_rbox3
from utils.training import Logger
from utils.vis import plot_box

height = 416
width  = 416

# TextBoxes++ with dense blocks and separable convolution
model = TBPP512_dense_separable(input_shape=(height, width, 3), softmax=False)
weights_path = './checkpoints/202003070004_dstbpp512fl_synthtext/weights.026.h5'
confidence_threshold = 0.45
plot_name = 'dstbpp512fl_sythtext'

model.load_weights(weights_path)
model.summary()

### saved_model_cli show --dir saved_model_512x512 --tag_set serve --signature_def serving_default

# The given SavedModel SignatureDef contains the following input(s):
#   inputs['input_1'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 512, 512, 3)
#       name: serving_default_input_1:0
# The given SavedModel SignatureDef contains the following output(s):
#   outputs['mbox_conf_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 76454, 2)
#       name: StatefulPartitionedCall:0
#   outputs['mbox_loc_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 76454, 4)
#       name: StatefulPartitionedCall:1
#   outputs['mbox_quad_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 76454, 8)
#       name: StatefulPartitionedCall:2
#   outputs['mbox_rbox_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 76454, 5)
#       name: StatefulPartitionedCall:3
# Method name is: tensorflow/serving/predict

### saved_model_cli show --dir saved_model_256x256 --tag_set serve --signature_def serving_default

# The given SavedModel SignatureDef contains the following input(s):
#   inputs['input_1'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 256, 256, 3)
#       name: serving_default_input_1:0
# The given SavedModel SignatureDef contains the following output(s):
#   outputs['mbox_conf_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 19124, 2)
#       name: StatefulPartitionedCall:0
#   outputs['mbox_loc_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 19124, 4)
#       name: StatefulPartitionedCall:1
#   outputs['mbox_quad_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 19124, 8)
#       name: StatefulPartitionedCall:2
#   outputs['mbox_rbox_final'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 19124, 5)
#       name: StatefulPartitionedCall:3
# Method name is: tensorflow/serving/predict



tf.saved_model.save(model, 'saved_model_{0}x{1}'.format(height, width))

# model.save('dstbpp512fl_sythtext_{0}}x{1}.h5'.format(height, width))
# open('dstbpp512fl_sythtext_{0}}x{1}_float32.json'.format(height, width), 'w').write(model.to_json())

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="dstbpp512fl_sythtext_{0}x{1}_float32.pb".format(height, width),
                    as_text=False)
print(".pb output complete! - dstbpp512fl_sythtext_{0}x{1}_float32.pb".format(height, width))

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('dstbpp512fl_sythtext_{0}x{1}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - dstbpp512fl_sythtext_{0}x{1}_float32.tflite".format(height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('dstbpp512fl_sythtext_{0}x{1}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - dstbpp512fl_sythtext_{0}x{1}_weight_quant.tflite".format(height, width))


def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image - 127.5
        image = image * 0.007843
        image = image[np.newaxis, :, :, :]
        yield [image]

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('dstbpp512fl_sythtext_{0}x{1}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - dstbpp512fl_sythtext_{0}x{1}_integer_quant.tflite".format(height, width))


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('dstbpp512fl_sythtext_{0}x{1}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print("Full Integer Quantization complete! - dstbpp512fl_sythtext_{0}x{1}_full_integer_quant.tflite".format(height, width))


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('dstbpp512fl_sythtext_{0}x{1}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - dstbpp512fl_sythtext_{0}x{1}_float16_quant.tflite".format(height, width))


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "dstbpp512fl_sythtext_{0}x{1}_full_integer_quant.tflite".format(height, width)])
# print(result)
