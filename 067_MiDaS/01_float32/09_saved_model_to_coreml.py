### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model', source='tensorflow')
mlmodel.save("midas_384x384_float32.mlmodel")

