
#import skimage.io  # bug. need to import this before tensorflow
#import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.0001
CONV_WEIGHT_STDDEV = 0.01
FC_WEIGHT_DECAY = 0.0001
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 8:
        lr = 0.01
    elif epoch < 24:
        lr = 0.001
    elif epoch < 36:
        lr = 0.0001
    elif epoch < 50:
        lr = 0.00001
    else:
        lr = 0.000001
    return {"lr":lr, "momentum":momentum}

def regularizer():
    return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)




def inference(inputs, training_mode):
    num_blocks=[2, 2, 2, 2]  # 6n+2 total weight layers will be used.
    use_bias=False # defaults to using batch norm
    use_bias = True
    num_classes = 1000
    is_training = training_mode
    bottleneck = False

    use_bias = True
    with tf.variable_scope('scale1'):
        #block_filters_internal = 16
        conv_filters_out = 64
        stride = 1
        x = conv(inputs, 7, stride, conv_filters_out)
        x = bn(x, use_bias, is_training)
        x = activation(x)

    with tf.variable_scope('scale2'):
        block_filters_internal = 64
        stride = 1
        x = stack(x, bottleneck, stride, num_blocks[0], use_bias, block_filters_internal, is_training)

    with tf.variable_scope('scale3'):
        block_filters_internal= 128
        stride = 2
        x = stack(x, bottleneck, stride, num_blocks[1], use_bias, block_filters_internal, is_training)

    with tf.variable_scope('scale4'):
        block_filters_internal= 256
        stride = 2
        x = stack(x, bottleneck, stride, num_blocks[2], use_bias, block_filters_internal, is_training)

    with tf.variable_scope('scale5'):
        block_filters_internal= 256
        stride = 2
        x = stack(x, bottleneck, stride, num_blocks[3], use_bias, block_filters_internal, is_training)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x, num_classes)
    return x



def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.scalar_summary('loss', loss_)

    return loss_


def stack(x, bottleneck, stride, num_blocks, use_bias, block_filters_internal, is_training):
    for n in range(num_blocks):
        s = stride if n == 0 else 1
        stride = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, bottleneck, stride, use_bias, block_filters_internal, is_training)
    return x


def block(x, bottleneck, stride, use_bias, block_filters_internal, is_training):
    filters_in = x.get_shape()[-1]

    m = 4 if bottleneck else 1
    filters_out = m * block_filters_internal
    conv_filters_out = block_filters_internal

    shortcut = x  # branch 1
    if bottleneck:
        with tf.variable_scope('a'):
            x = conv(x, 1, stride, conv_filters_out)
            x = bn(x, use_bias, is_training)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, 1, stride, conv_filters_out)
            x = bn(x, use_bias, is_training)
            x = activation(x)

        with tf.variable_scope('c'):
            x = conv(x, 1, 1, filters_out)
            x = bn(x, use_bias, is_training)
    else:
        with tf.variable_scope('A'):
            x = conv(x, 3, stride, conv_filters_out)
            x = bn(x, use_bias, is_training)
            x = activation(x)

        with tf.variable_scope('B'):
            x = conv(x, 3, 1, filters_out)
            x = bn(x, use_bias, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or stride != 1:
            shortcut = conv(shortcut, 1, stride, filters_out)
            shortcut = bn(shortcut, use_bias, is_training)

    return activation(x + shortcut)


def bn(x, use_bias, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = _get_variable('bias', params_shape, initializer=tf.zeros_initializer())
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=True)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=True)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    #num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  trainable=True):

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    #dtype='float'
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype='float',
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, ksize, stride, filters_out):
    #ksize = c['ksize']
    #stride = c['stride']
    #filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
