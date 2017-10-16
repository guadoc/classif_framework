import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops



def bn(x, is_training):
    BN_DECAY = 0.9#0.9997
    BN_EPSILON = 0.00001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
    moving_mean     = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)
    tf.add_to_collection("variable_to_save", moving_mean)
    tf.add_to_collection("variable_to_save", moving_variance)
    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    update_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY), lambda: moving_mean)
    update_moving_variance = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY), lambda: moving_variance)

    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON), [beta, gamma]
