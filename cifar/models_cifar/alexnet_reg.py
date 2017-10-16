import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
import numpy as np
from layers.activation import relu
from layers.pooling import max_pool_2D

REG_COEF = 0.8
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01

CONV_WEIGHT_DECAY = 0.00005
FC_WEIGHT_DECAY= 0.00001
#da
#CONV = 0.000005 < 0.00005
#FC = 0.000001 < 0.000001
#sans da
#CONV = 0.000005 __last test = 0.000001
#FC   = 0.000001 __last test = 0.000005


def optim_param_schedule(monitor):
    print("CONV_WEIGHT_DECAY: "+str(CONV_WEIGHT_DECAY))
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 120:
        lr = 0.01
    elif epoch < 180:
        lr = 0.001
    else:
        lr = 0.00001
    lr = 0.05*math.pow(0.98, epoch-1)
    print("lr: "+str(lr)+", momentum: "+str(momentum) )
    return {"lr":lr, "momentum":momentum}

def optim_param_schedule_(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 3:
        lr=0
    else:
        lr = 0.001# peut être 0.0001
    print("lr: "+str(lr))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 5
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg')[0], tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def regul(outputs):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    p_a1 = tf.reduce_mean(p_mode_1, [0])
    p_a2 = tf.reduce_mean(p_mode_2, [0])
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_a1), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.divide(tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), p_a2), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_1) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_2) ), p_mode_2) ,[0])  )
    return tf.add(var_1, var_2)


def regul_(outputs):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.zeros_initializer(), trainable=False)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(outputs, [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(tf.reduce_mean( tf.square( tf.subtract(outputs, mean_1) ) ,[0])  )
    return var_1


def conv_me(x, ksize, stride, filters_out):
    collection = tf.get_variable_scope().name
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights =tf.get_variable('weights', shape=shape, initializer=initializer)
    tf.add_to_collection(collection+"_variables", weights)
    outputs = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    shape_output = outputs.get_shape()
    sizeX = shape_output[1]
    sizeY = shape_output[2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    outs = outputs[:, indexX, indexY, :]
    reg = regul(outs)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
    return outputs


def fc_me(x, num_units_out):
    collection = tf.get_variable_scope().name

    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    reg = regul(outputs)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_5'):
        n_out = 32
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = relu(x)
        x = max_pool_2D(x, 2, 2)

    with tf.variable_scope('layer_4'):
        n_out = 64
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = relu(x)
        x = max_pool_2D(x, 2, 2)

    with tf.variable_scope('layer_3'):
        n_out = 64
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = relu(x)
        x = max_pool_2D(x, 2, 2)
        x = tf.reshape(x, [-1, 4*4*n_out])
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.2), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_2'):
        n_out = 1000
        x = fc_me(x, n_out)
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.3), lambda: tf.nn.dropout(x, 1))
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2], collections=['per_batch'])
        x = relu(x)
    with tf.variable_scope('layer_1'):
        outputs = fc_me(x, 10)
        #tr_activation_summary = tf.summary.histogram('activation1', outputs[:, 2], collections=['per_batch'])

    return outputs, tf.nn.softmax(outputs)
