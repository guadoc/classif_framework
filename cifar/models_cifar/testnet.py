import tensorflow as tf
from layers.activation import relu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block
from layers.normalization import bn
from layers.regularization import weight_decay, var, shade, shade_conv
#import math
import numpy as np
REG_COEF = 0.8
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01

CONV_WEIGHT_DECAY = 0#0.0001
FC_WEIGHT_DECAY= 0#0.0005
#CONV_WEIGHT_DECAY = 0.000000005
#FC_WEIGHT_DECAY= 0.00000001

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 120:
        lr = 0.01
    elif epoch < 180:
        lr = 0.001
    else:
        lr = 0.00001
    return {"lr":lr, "momentum":momentum}

def layer_regularizer():
    n_layers = 5
    regs = []
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_5'):
        field = 'conv_1'
        n_out = 16
        x, params = conv_2D(x, 5, 1, n_out, conv_init, field, True)
        weight_decay(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
        #x = relu(x)
        x = max_pool_2D(x, 3, 2)

    with tf.variable_scope('layer_4'):
        field = 'res1'
        ksizes = [3,3]
        strides = [1, 1]
        filters_out = [32, 32]
        x, params = residual_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
        #x, params = conv_2D(x, 5, 1, n_out, conv_init, field, True)
        weight_decay(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
        #tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
        x = max_pool_2D(x, 3, 2)

    with tf.variable_scope('layer_3'):
        n_out = 64
        field = 'conv_3'
        x, params = conv_2D(x, 5, 1, n_out, conv_init, field)
        weight_decay(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
        x = bn(x, training_mode, field)
        x = relu(x)
        x = max_pool_2D(x, 3, 2)

        x = tf.reshape(x, [-1, 4*4*n_out])

    with tf.variable_scope('layer_2'):
        field = 'fc1'
        n_out = 1000
        x, params = fc(x, n_out, fc_init, field)
        weight_decay(x, params, tf.get_variable_scope().name, field, FC_WEIGHT_DECAY)
        x = relu(x)


    with tf.variable_scope('layer_1'):
        field = 'fc2'
        outputs, params = fc(x, 10, fc_init, field)
        weight_decay(outputs, params, tf.get_variable_scope().name, field, FC_WEIGHT_DECAY)


    return outputs, outputs
