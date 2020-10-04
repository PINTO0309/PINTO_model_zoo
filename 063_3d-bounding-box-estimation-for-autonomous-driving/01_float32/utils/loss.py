import tensorflow as tf


def orientation_loss(y_true, y_pred):
    """
    input: y_true -- (batch_size, bin, 2) ground truth orientation value in cos and sin form.
           y_pred -- (batch_size, bin ,2) estimated orientation value from the ConvNet
    output: loss -- loss values for orientation
    """

    # sin^2 + cons^2
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    # check which bin valid
    anchors = tf.greater(anchors, tf.constant(0.5))
    # add valid bin
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    # cos(true)cos(estimate) + sin(true)sin(estimate)
    loss = (y_true[:, : ,0] * y_pred[:, :, 0] + y_true[:, :, 1]*y_pred[:, :, 1])
    # the mean value in each bin
    loss = tf.reduce_sum(loss, axis=1) / anchors
    # sum the value at each bin
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss

    return loss

