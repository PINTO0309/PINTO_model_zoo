### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_kitti_192x640', source='tensorflow')
mlmodel.save("footprints_kitti_192x640_float32.mlmodel")
