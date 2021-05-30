import tensorflow as tf

model = tf.keras.models.load_model('weights/model_unet.h5')
tf.saved_model.save(model, 'saved_model')