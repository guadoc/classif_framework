import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import math

REG_COEF = 0.9
FC_WEIGHT_STDDEV=0.05
CONV_WEIGHT_STDDEV=0.05
CONV_WEIGHT_DECAY = 0#0.00001
FC_WEIGHT_DECAY= 0#0.00001

UPDATE_OPS_COLLECTION = 'resnet_update_ops'
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001

def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.1*math.pow(0.95, monitor.epoch-1)
    print("lr: "+str(lr)+", momentum: "+str(momentum))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 12
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, dtype='float')
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
    collection = tf.get_variable_scope().name
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    reg = tf.multiply(tf.nn.l2_loss(weights), CONV_WEIGHT_DECAY)
    tf.add_to_collection(collection+'_reg', reg)
    return tf.nn.xw_plus_b(x, weights, biases)


def bn(x, use_bias, is_training, ksize):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    if use_bias:
        bias = tf.get_variable('bias'+str(ksize), params_shape, initializer=tf.zeros_initializer(), trainable=True)
        return x + bias
    axis = list(range(len(x_shape) - 1))
    beta = tf.get_variable('beta'+str(ksize), params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable('gamma'+str(ksize), params_shape, initializer=tf.ones_initializer())
    moving_mean = tf.get_variable('moving_mean'+str(ksize), params_shape, initializer=tf.zeros_initializer(), trainable=True)
    moving_variance = tf.get_variable('moving_variance'+str(ksize), params_shape, initializer=tf.ones_initializer(), trainable=True)
    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average( moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    # mean, variance = control_flow_ops.cond(
    #     is_training, lambda: (mean, variance),
    #     lambda: (update_moving_mean, update_moving_variance))

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (update_moving_mean, update_moving_variance),
        lambda: (moving_mean, moving_variance))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

def conv(x, ksize, stride, filters_out, is_training):
    filters_in = x.get_shape()[-1]
    collection = tf.get_variable_scope().name
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    #weights = _get_variable('weights', collection = tf.get_variable_scope().name, shape=shape, initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
    weights = tf.get_variable('weights'+str(ksize), shape=shape, initializer=initializer, dtype='float')
    tf.add_to_collection(collection+"_variables", weights)
    reg = tf.multiply(tf.nn.l2_loss(weights), CONV_WEIGHT_DECAY)
    tf.add_to_collection(collection+'_reg', reg)
    out = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    out = bn(out, True, is_training, ksize)
    return activation(out)



def inception(x, f1_1, f3_3, is_training):
    x1_1 = conv(x, 1, 1, f1_1, is_training)
    x3_3 = conv(x, 3, 1, f3_3, is_training)
    return tf.concat([x1_1, x3_3], -1)


def downsample(x, filter_conv, is_training):
    x_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    x3_3 = conv(x, 3, 2, filter_conv,is_training)
    return tf.concat([x_pool, x3_3], -1)



def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_12'):
        n_out = 96
        x = conv(x, 3, 1, n_out, training_mode)

    with tf.variable_scope('layer_11'):
        x = inception(x, 32, 32, training_mode)

    with tf.variable_scope('layer_10'):
        x = inception(x, 32, 48, training_mode)

    with tf.variable_scope('layer_9'):
        x = downsample(x, 80, training_mode)

    with tf.variable_scope('layer_8'):
        x = inception(x, 112, 48, training_mode)

    with tf.variable_scope('layer_7'):
        x = inception(x, 96, 64, training_mode)

    with tf.variable_scope('layer_6'):
        x = inception(x, 80, 80, training_mode)

    with tf.variable_scope('layer_5'):
        x = inception(x, 48, 96, training_mode)

    with tf.variable_scope('layer_4'):
        x = downsample(x, 96, training_mode)

    with tf.variable_scope('layer_3'):
        x = inception(x, 176, 160, training_mode)

    with tf.variable_scope('layer_2'):
        x = inception(x, 176, 160, training_mode)
        #x = tf.Print(x, [tf.shape(x)])
        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, [-1, 336])

    with tf.variable_scope('layer_1'):
        outputs = fc(x, 10)

    return outputs, outputs
