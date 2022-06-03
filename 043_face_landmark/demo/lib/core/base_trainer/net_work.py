#-*-coding:utf-8-*-


import tensorflow as tf


import time
import os

from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

from lib.core.model.shufflenet.simpleface import calculate_loss
from lib.helper.logger import logger


class Train(object):
  """Train class.
  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, epochs, enable_function, model, batch_size, strategy):
    self.epochs = epochs
    self.batch_size = batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.enable_function = enable_function
    self.strategy = strategy

    if 'Adam' in cfg.TRAIN.opt:
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.lr_decay_every_epoch[0])
    else:
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.TRAIN.lr_decay_every_epoch[0],momentum=0.9)

    self.model = model




    ###control vars
    self.iter_num=0

    self.lr_decay_every_epoch =cfg.TRAIN.lr_decay_every_epoch
    self.lr_val_every_epoch = cfg.TRAIN.lr_value_every_epoch






  def decay(self, epoch):

    if epoch < self.lr_decay_every_epoch[0]:
      return self.lr_val_every_epoch[0]
    if epoch >= self.lr_decay_every_epoch[0] and epoch < self.lr_decay_every_epoch[1]:
      return self.lr_val_every_epoch[1]

    if epoch >= self.lr_decay_every_epoch[1] and epoch < self.lr_decay_every_epoch[2]:
      return self.lr_val_every_epoch[2]
    if epoch >= self.lr_decay_every_epoch[2] and epoch < self.lr_decay_every_epoch[3]:
      return self.lr_val_every_epoch[3]
    if epoch >= self.lr_decay_every_epoch[3] and epoch < self.lr_decay_every_epoch[4]:
      return self.lr_val_every_epoch[4]
    if epoch >= self.lr_decay_every_epoch[4] and epoch < self.lr_decay_every_epoch[5]:
      return self.lr_val_every_epoch[5]
    if epoch >= self.lr_decay_every_epoch[5]:
      return self.lr_val_every_epoch[6]





  def weight_decay_loss(self,):

    regularization_loss=0.

    for variable in self.model.trainable_variables:
      if 'kernel' in variable.name:
        regularization_loss+=tf.math.reduce_sum(tf.math.square(variable))

    return regularization_loss*self.l2_regularization*0.5

  def compute_loss(self, predictions, label,training=False):

    loss = tf.reduce_sum(calculate_loss(predictions, label))
    if training:
      loss += (self.weight_decay_loss() * 1. / self.strategy.num_replicas_in_sync)
    return loss


  def train_step(self, inputs):
    """One train step.
    Args:
      inputs: one batch input.
    Returns:
      loss: Scaled loss.
    """

    image, label = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)

      loss = self.compute_loss(predictions,label,training=True)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    gradients = [(tf.clip_by_value(grad, -5.0, 5.0))
                 for grad in gradients]
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    return loss

  def test_step(self, inputs):
    """One test step.
    Args:
      inputs: one batch input.
    """

    image, label = inputs

    predictions = self.model(image,training=False)

    unscaled_test_loss = self.compute_loss(predictions,label,training=False)

    return unscaled_test_loss


  def custom_loop(self, train_dist_dataset, test_dist_dataset,
                  strategy):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(ds,epoch_num):
      total_loss = 0.0
      num_train_batches = 0.0
      for one_batch in ds:

        start=time.time()
        per_replica_loss = strategy.experimental_run_v2(
            self.train_step, args=(one_batch,))
        current_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss += current_loss
        num_train_batches += 1
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:
          logger.info('epoch_num: %d, '
                      'iter_num: %d, '
                      'loss_value: %.6f,  '
                      'speed: %d images/sec ' % (epoch_num,
                                                 self.iter_num,
                                                 current_loss,
                                                 images_per_sec))

      return total_loss, num_train_batches

    def distributed_test_epoch(ds,epoch_num):
      total_loss=0.
      num_test_batches = 0.0
      for one_batch in ds:
        per_replica_loss=strategy.experimental_run_v2(
            self.test_step, args=(one_batch,))

        current_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss+=current_loss
        num_test_batches += 1
      return total_loss, num_test_batches

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      distributed_test_epoch = tf.function(distributed_test_epoch)

    for epoch in range(self.epochs):

      start=time.time()
      self.optimizer.learning_rate = self.decay(epoch)

      train_total_loss, num_train_batches = distributed_train_epoch(
          train_dist_dataset,epoch)
      test_total_loss, num_test_batches = distributed_test_epoch(
          test_dist_dataset,epoch)



      time_consume_per_epoch=time.time()-start
      training_massage = 'Epoch: %d, ' \
                         'Train Loss: %.6f, ' \
                         'Test Loss: %.6f '\
                         'Time consume: %.2f'%(epoch,
                                               train_total_loss / num_train_batches,
                                               test_total_loss / num_test_batches,
                                               time_consume_per_epoch)

      logger.info(training_massage)


      #### save the model every end of epoch
      current_model_saved_name=os.path.join(cfg.MODEL.model_path,
                                            'epoch_%d_val_loss%.6f'%(epoch,test_total_loss / num_test_batches))

      logger.info('A model saved to %s' % current_model_saved_name)

      if not os.access(cfg.MODEL.model_path,os.F_OK):
        os.mkdir(cfg.MODEL.model_path)

      tf.saved_model.save(self.model,current_model_saved_name)



    return (train_total_loss / num_train_batches,
            test_total_loss / num_test_batches)





