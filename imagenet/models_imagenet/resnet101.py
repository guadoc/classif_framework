
import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.random_normal_initializer
fc_init = tf.random_normal_initializer(0.01)#tf.uniform_unit_scaling_initializer(factor=1.0)



FC_WEIGHT_DECAY = 0.0001
CONV_WEIGHT_DECAY = 0.0001

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 40:
        lr = 0.01
    elif epoch < 70:
        lr = 0.001
    else:
        lr = 0.00005
    lr = 0.00001
    #momentum = 0
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 1
    regs = []
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def block(x, n_block, strides, f_out, activation, training_mode, field):
    regularization = weight_decay
    for i in range(n_block):
        stride = 1
        if i == 0: stride=strides
        ksizes = [1,3,1]
        strides = [stride, 1, 1]
        filters_out = [f_out, f_out, 4*f_out]
        field_ = field+'res'+str(i+1)
        x, params = residual_block(x, ksizes, strides, filters_out, conv_init(0.1), activation, field_, training_mode)
        regularization(x, params, tf.get_variable_scope().name, field_, CONV_WEIGHT_DECAY)
    return x


def inference(inputs, training_mode):
    x = inputs
    regularization = shade_conv#weight_decay
    #x = tf.Print(x, ["mean", tf.reduce_mean(x)])
    N_LAYER = 0
    IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]    
    red, green, blue = tf.split(x, 3, 3)
    x = tf.concat([blue, green, red], 3)
    x = x - IMAGENET_MEAN_BGR

    N_LAYER=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'conv1'
        n_out = 64
        x, params = conv_2D(x, 7, 2, n_out, conv_init(0.1), field, False)
        regularization(x, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)
        x = bn(x, training_mode, 'bn1')
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


    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    N_LAYER+=1
    with tf.variable_scope('layer_'+str(N_LAYER)):
        field = 'fc1'
        n_out = 1000
        outputs, params = fc(x, n_out, fc_init, field)
        regularization(outputs, params, tf.get_variable_scope().name, field, CONV_WEIGHT_DECAY)


    print('ResNet with '+str(N_LAYER) + ' scales')
    return outputs, outputs
