### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_age', source='tensorflow')
mlmodel.save("age_gender_recognition_62x62_float32.mlmodel")

