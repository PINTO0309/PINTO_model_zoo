import numpy as np
import tensorflow as tf
optimizer = tf.keras.optimizers
callbacks = tf.keras.callbacks
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from data_processing.KITTI_dataloader import KITTILoader
from data_processing.preprocessing import orientation_confidence_flip

from utils.data_generation import data_gen
from utils.loss import orientation_loss

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2 as nn

def train():
    KITTI_train_gen = KITTILoader(subset='training')
    dim_avg, dim_cnt = KITTI_train_gen.get_average_dimension()

    new_data = orientation_confidence_flip(KITTI_train_gen.image_data, dim_avg)

    model = nn.network()
    # model.load_weights('3dbox_weights_mob.hdf5')

    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint = callbacks.ModelCheckpoint('3dbox_mbnv2_{}x{}_float32.hdf5'.format(cfg().norm_h, cfg().norm_w), monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    tensorboard = callbacks.TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=False)

    all_examples = len(new_data)
    trv_split = int(cfg().split * all_examples) # train val split

    train_gen = data_gen(new_data[: trv_split])
    valid_gen = data_gen(new_data[trv_split : all_examples])

    train_num = int(np.ceil(trv_split / cfg().batch_size))
    valid_num = int(np.ceil((all_examples - trv_split) / cfg().batch_size))

    # choose the minimizer to be sgd
    minimizer = optimizer.SGD(lr=0.0001, momentum = 0.9)

    # multi task learning
    model.compile(optimizer=minimizer,  #minimizer,
                  loss={'dimensions': 'mean_squared_error', 'orientation': orientation_loss, 'confidence': 'binary_crossentropy'},
                  loss_weights={'dimensions': 1., 'orientation': 10., 'confidence': 5.})
    # d:0.0088 o:0.0042, c:0.0098

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_num,
                        epochs=500,
                        verbose=1,
                        validation_data=valid_gen,
                        validation_steps=valid_num,
                        shuffle=True,
                        callbacks=[early_stop, checkpoint, tensorboard],
                        max_queue_size=3)

    tf.saved_model.save(model, 'saved_model_{}x{}'.format(cfg().norm_h, cfg().norm_w))
    model.save('3dbox_mbnv2_{}x{}_float32.h5'.format(cfg().norm_h, cfg().norm_w))

    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=".",
                        name="3dbox_mbnv2_{}x{}_float32.pb".format(cfg().norm_h, cfg().norm_w),
                        as_text=False)

if __name__ == '__main__':
    train()