import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('defocus_deblurring_dp_l5_s512_f0.7_d0.4.hdf5')
tf.saved_model.save(model, 'saved_model')

'''
$ saved_model_cli show --dir saved_model --tag_set serve --signature_def serving_default

The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 512, 512, 6)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['conv2d_24'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 512, 512, 3)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
'''