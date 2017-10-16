

import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block_nb, shake_block
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.truncated_normal_initializer(stddev=0.01)
fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)


CONV_WEIGHT_DECAY = 0.005   #0.0001 of 0.0005
FC_WEIGHT_DECAY = 0.005

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.08*math.pow(0.99, monitor.epoch-1)
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 14
    regs = []
    print('Layer number to regularize :'+ str(n_layers))
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs



def inference(inputs, training_mode):
    x = inputs
    N_LAYER = 0
    N = 4
    k = 10
    regul_conv = shade_conv#weight_decay
    regul_fc = shade#weight_decay

    N_LAYER+=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'conv_1'
        n_out = 16
        x, params = conv_2D(x, 3, 1, n_out, conv_init, field, True)
        x = bn(x, training_mode, 'bn1')
        regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)

    for i in range(N):
        N_LAYER+=1
        with tf.variable_scope('layer_'+str(N_LAYER)):
            field = 'shake'+str(N_LAYER)
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [16, 16]
            x, params = shake_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
            regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)


    N_LAYER+=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'shake'+str(N_LAYER)
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [32, 32]
        x, params = shake_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
        regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('layer_'+str(N_LAYER)):
            field = 'shake'+str(N_LAYER)
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [32, 32]
            x, params = shake_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
            regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)


    N_LAYER+=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'shake'+str(N_LAYER)
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [64, 64]
        x, params = shake_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
        regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('layer_'+str(N_LAYER)):
            field = 'shake'+str(N_LAYER)
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [64, 64]
            x, params = shake_block(x, ksizes, strides, filters_out, conv_init, relu, field, training_mode)
            regul_conv(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)

    x = relu(x)
    #x = tf.Print(x, [tf.shape(x)])
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.9), lambda: tf.nn.dropout(x, 1))
    N_LAYER+=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'fc1'
        n_out = 10
        outputs, params = fc(x, n_out, fc_init, field)
        regul_fc(outputs, params, tf.get_variable_scope().name, field, FC_WEIGHT_DECAY)
    print('ResNet with '+str(N_LAYER) + ' layers')

    return outputs, outputs
