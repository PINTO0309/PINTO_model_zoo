### tensorflow-gpu==2.2.0
### onnx==1.7.0
### onnx2keras==0.0.21
### https://github.com/amir-abdi/keras_to_tensorflow.git

import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
import shutil
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

onnx_model = onnx.load('dbface.onnx')
k_model = onnx_to_keras(onnx_model=onnx_model, input_names=['x'])

shutil.rmtree('saved_model', ignore_errors=True)
tf.saved_model.save(k_model, 'saved_model')

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: k_model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(k_model.inputs[0].shape, k_model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=".", name="dbface.pb", as_text=False)

k_model.save('dbface.h5', include_optimizer=False)