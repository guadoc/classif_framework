import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block_nb
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.truncated_normal_initializer(stddev=0.01)
fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)


CONV_WEIGHT_DECAY = 0.#005   #0.0001 of 0.0005
FC_WEIGHT_DECAY = 0.#001

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    #lr = 0.1*math.pow(0.95, monitor.epoch-1)
    if monitor.epoch < 40:
        lr = 0.1
    elif monitor.epoch < 80:
        lr = 0.01
    elif monitor.epoch < 200:
        lr = 0.001
    elif monitor.epoch < 400:
        lr = 0.0001
    else:
        lr = 0.00001
    lr = 0.04*math.pow(0.99, monitor.epoch-1)
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
    N = 9#4
    k = 1#10
    regul_conv = shade_conv#weight_decay
    regul_fc = shade#weight_decay

    N_LAYER+=1
    layer = 0
    with tf.variable_scope('layer_'+str(N_LAYER)):
        n_out = 16
        x, params = conv_2D(x, 3, 1, n_out, conv_init, use_biases=True, padding='VALID')
        x, params_bn = bn(x, training_mode)
        x = relu(x)
        layer+=1
        regul_conv(x, [params, params_bn], 'layer_'+str(layer), CONV_WEIGHT_DECAY)


    for i in range(N):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [16*k, 16*k]
            x, params = residual_block_nb(x, ksizes, strides, filters_out, conv_init, relu, training_mode)
            layer+=1
            regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)


    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [32*k, 32*k]
        x, params = residual_block_nb(x, ksizes, strides, filters_out, conv_init, relu, training_mode)
        layer+=1
        regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)

    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [32*k, 32*k]
            x, params = residual_block_nb(x, ksizes, strides, filters_out, conv_init, relu, training_mode)
            layer+=1
            regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)


    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [64*k, 64*k]
        x, params = residual_block_nb(x, ksizes, strides, filters_out, conv_init, relu, training_mode)
        layer+=1
        regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)

    #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.9), lambda: tf.nn.dropout(x, 1))
    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [64*k, 64*k]
            x, params = residual_block_nb(x, ksizes, strides, filters_out, conv_init, relu, training_mode)
            layer+=1
            regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)


    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.9), lambda: tf.nn.dropout(x, 1))
    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):        
        n_out = 100
        outputs, params = fc(x, n_out, fc_init)
        layer+=1
        regul_fc(outputs, params, 'layer_'+str(layer), FC_WEIGHT_DECAY)
    print('ResNet with '+str(layer) + ' layers')

    return outputs, 0
