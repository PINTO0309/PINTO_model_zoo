### sudo pip3 install onnx2keras tf-nightly
### tf-nightly-2.2.0-dev20200502

import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
import shutil

onnx_model = onnx.load('human-pose-estimation-3d-0001.onnx')
k_model = onnx_to_keras(onnx_model=onnx_model, input_names=['data'], change_ordering=True)

shutil.rmtree('saved_model', ignore_errors=True)
tf.saved_model.save(k_model, 'saved_model')

"""
$ saved_model_cli show --dir saved_model --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['data'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 256, 448, 3)
        name: serving_default_data:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['features'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 32, 56, 57)
        name: StatefulPartitionedCall:0
    outputs['heatmaps'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 32, 56, 19)
        name: StatefulPartitionedCall:1
    outputs['pafs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 32, 56, 38)
        name: StatefulPartitionedCall:2
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          data: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='data')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          data: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='data')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          data: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='data')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          data: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='data')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          data: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='data')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 256, 448, 3), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
"""