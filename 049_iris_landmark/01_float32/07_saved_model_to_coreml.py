### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model', source='tensorflow')
mlmodel.save("iris_landmark_64x64_float32.mlmodel")
