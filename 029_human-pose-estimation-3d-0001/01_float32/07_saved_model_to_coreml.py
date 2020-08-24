### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model', source='tensorflow')
mlmodel.save("human-pose-estimation-3d-0001_256x448_float32.mlmodel")
