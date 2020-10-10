### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_448x448', source='tensorflow')
mlmodel.save("resnet34_3inputs_448x448_float32.mlmodel")
