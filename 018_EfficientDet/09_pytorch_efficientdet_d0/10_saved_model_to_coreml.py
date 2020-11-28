### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_480x640', source='tensorflow')
mlmodel.save("retinanet-9_480x640_float32.mlmodel")

