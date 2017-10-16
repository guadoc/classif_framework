
import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D#, residual_block
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.random_normal_initializer
fc_init = tf.random_normal_initializer(0.01)#tf.uniform_unit_scaling_initializer(factor=1.0)


CONV_WEIGHT_DECAY = 0.1#0.000001#0.0001

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 5:
        lr = 0.00001
    elif epoch < 12:
        lr = 0.000001
    else:
        lr = 0.0000002
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 1#6
    regs = []
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs



def residual_block_weldone(x, ksizes, strides, filters_out, initializer, activation, l, is_training):
    filters_in = x.get_shape()[-1]
    tot_stride = 0
    params = []
    shortcut = x       
    with tf.variable_scope(l+'/conv1'):
        x, w = conv_2D(x, ksizes[0], strides[0], filters_out[0], initializer)
        params+= w
        tot_stride += strides[0]
        x = bn(x, is_training)
        x = activation(x)    
    with tf.variable_scope(l+'/conv2'):        
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")        
        x, w = conv_2D(x, ksizes[1], strides[1], filters_out[1], initializer)
        #params+= w
        tot_stride += strides[1]
        x = bn(x, is_training)
        x = activation(x)    
    with tf.variable_scope(l+'/conv3'):          
        x, w = conv_2D(x, ksizes[2], strides[2], filters_out[2], initializer)
        #params+= w
        tot_stride += strides[2]
        x = bn(x, is_training)

    if filters_out[-1] != filters_in or tot_stride > len(filters_out):        
        with tf.variable_scope(l+'/shortcut'):
            shortcut, w = conv_2D(shortcut, 1, tot_stride - len(filters_out)+1, filters_out[-1], initializer)        
            shortcut = bn(shortcut, is_training)
        #params+= w
    return activation(x + shortcut), params


def block(x, n_block, strides, f_out, activation, training_mode, field):
    regularization = shade_conv#weight_decay
    for i in range(n_block):
        stride = 1
        if i == 0: stride=strides
        ksizes = [1,3,1]
        strides = [stride, 1, 1]
        filters_out = [f_out, f_out, 4*f_out]
        field_ = field+'res'+str(i+1)
        x, params = residual_block_weldone(x, ksizes, strides, filters_out, conv_init(0.1), activation, field_, training_mode)
        #regularization(x, params, tf.get_variable_scope().name, field_, CONV_WEIGHT_DECAY)
    return x


def inference(inputs, training_mode):
    x = inputs
    regularization = weight_decay#shade_conv

    N_LAYER = 0
    IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
    red, green, blue = tf.split(x, 3, 3)
    x = tf.concat([blue, green, red], 3)
    x = x - IMAGENET_MEAN_BGR

    N_LAYER=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'conv1'
        n_out = 64
        x, params = conv_2D(x, 7, 2, n_out, conv_init(0.1), use_biases=False)
        regularization(x, params, 'layer_1', CONV_WEIGHT_DECAY)
        x = bn(x, training_mode)
        x = relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    N_LAYER=2
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = ''
        f_out = 64
        stride = 1
        x = block(x, 3, stride, f_out, relu, training_mode, field)

    N_LAYER=3
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = ''
        f_out = 128
        stride = 2
        x = block(x, 4, stride, f_out, relu, training_mode, field)

    N_LAYER=4
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = ''
        f_out = 256
        stride = 2
        x = block(x, 23, stride, f_out, relu, training_mode, field)

    N_LAYER=5
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = ''
        f_out = 512
        stride = 2
        x = block(x, 3, stride, f_out, relu, training_mode, field)

    N_LAYER=6
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'conv1'
        f_out = 1000
        stride = 1
        x, params = conv_2D(x, 1, 1, f_out, conv_init(0.1), use_biases=True)
        #regularization(x, params, ', CONV_WEIGHT_DECAY)


    #x = tf.reshape(x, [- 1, 14*14, 1000])
    #x = tf.transpose(x, [0, 2, 1])
    #sorted_val, ind = tf.nn.top_k(  x,    k=196,    sorted=True,    name=None)
    #outputs = (tf.reduce_sum(sorted_val[:, :, 0:50], axis=2) + tf.reduce_sum(sorted_val[:, :, 146:], axis=2))/50
    outputs = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    #outputs = tf.reduce_mean(x, axis=2)

    print('ResNet with '+str(N_LAYER) + ' scales')
    return outputs, tf.ones([1, 1])
