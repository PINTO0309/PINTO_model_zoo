### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_256x256', source='tensorflow')
mlmodel.save("u2netp_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_320x320', source='tensorflow')
mlmodel.save("u2netp_320x320_float32.mlmodel")

mlmodel = ct.convert('saved_model_480x640', source='tensorflow')
mlmodel.save("u2netp_480x640_float32.mlmodel")