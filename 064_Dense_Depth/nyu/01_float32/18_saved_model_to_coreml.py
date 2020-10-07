### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_nyu_480x640', source='tensorflow')
mlmodel.save("dense_depth_nyu_480x640_float32.mlmodel")
