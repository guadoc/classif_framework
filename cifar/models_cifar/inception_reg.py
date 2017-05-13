import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import math
import numpy as np
REG_COEF = 0.99
#da
#CONV = 0.00005
#FC = 0.00001
#sans da
#CONV = 0.000005
#FC = 0.000001
CONV_WEIGHT_DECAY = 0.00005
FC_WEIGHT_DECAY= 0.00001

FC_WEIGHT_STDDEV=0.05
CONV_WEIGHT_STDDEV=0.05

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
    print("lr: "+str(lr))
    return {"lr":lr, "momentum":momentum}

def regularizer():
    return tf.get_collection('reg')

def layer_regularizer():
    print("decay: "+str(CONV_WEIGHT_DECAY))
    n_layers = 12
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def regul(outputs, ksize):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1'+str(ksize), [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2'+str(ksize), trainable=False)
    p_mode_1 = tf.sigmoid(outputs)#tf.cast(tf.greater(outputs, tf.zeros_like(outputs)), tf.float32)#
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_1) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_2) ), p_mode_2) ,[0])  )
    return tf.add(var_1, var_2)

def fc_e(x, num_units_out):
    collection = tf.get_variable_scope().name
    num_units_in = x.get_shape().as_list()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases  = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    reg = regul(outputs, 0)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs

def conv_e(x, ksize, stride, filters_out, is_training):
    collection = tf.get_variable_scope().name
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights =tf.get_variable('weights'+str(ksize), shape=shape, initializer=initializer)
    tf.add_to_collection(collection+"_variables", weights)
    outputs = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    shape_output = outputs.get_shape()
    sizeX = shape_output[1]
    sizeY = shape_output[2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    outs = outputs[:, indexX, indexY, :]
    reg = regul(outs, ksize)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
    outputs = bn(outputs, True, is_training, ksize)
    return activation(outputs)


def bn(x, use_bias, is_training, ksize):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    if use_bias:
        bias = tf.get_variable('bias'+str(ksize), params_shape, initializer=tf.zeros_initializer(), trainable=True)
        collection = tf.get_variable_scope().name
        tf.add_to_collection(collection+"_variables", bias)
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



def inception(x, f1_1, f3_3, is_training):
    x1_1 = conv_e(x, 1, 1, f1_1, is_training)
    x3_3 = conv_e(x, 3, 1, f3_3, is_training)
    return tf.concat([x1_1, x3_3], -1)


def downsample(x, filter_conv, is_training):
    x_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    x3_3 = conv_e(x, 3, 2, filter_conv,is_training)
    return tf.concat([x_pool, x3_3], -1)



def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_12'):
        n_out = 96
        x = conv_e(x, 3, 1, n_out, training_mode)

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
        outputs = fc_e(x, 10)

    return outputs, outputs
