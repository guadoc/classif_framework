
#import skimage.io  # bug. need to import this before tensorflow
#import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf


import datetime
import numpy as np
import os
import time

CONV_WEIGHT_DECAY = 0.0005
CONV_WEIGHT_STDDEV = 0.01
FC_WEIGHT_DECAY = 0.0005
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")
activation = tf.nn.relu

def regularizer():
    return tf.get_collection('regularizer')

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 18:
        lr = 0.01
    elif epoch < 40:
        lr = 0.001
    elif epoch < 80:
        lr = 0.0001
    else:
        lr = 0.00001
    return {"lr":lr, "momentum":momentum}

def inference(inputs, training_mode):
    num_classes = 1000
    mul=2
    is_training = training_mode
    x1 = convs(inputs, '1')
    x2 = convs(inputs, '2')

    x = tf.concat([x1,x2], 1)
    x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.5), lambda: tf.nn.dropout(x, 1))
    with tf.variable_scope('fc1'):
        x = fc(x, 2048*mul)
        x = activation(x)

    x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.5), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('fc2'):
        x = fc(x, 2048*mul)
    x = activation(x)
    with tf.variable_scope('fc3'):
        x = fc(x, num_classes)
    return x


def convs(inputs_batch, scope):
    with tf.variable_scope('A'+scope):
        x = conv(inputs_batch, 7, 4, 48)
    x = activation(x)
    x = _max_pool(x, ksize=3, stride=2)
    with tf.variable_scope('B'+scope):
        x = conv(x, 3, 1, 128)
    x = activation(x)
    x = _max_pool(x, ksize=3, stride=2)
    with tf.variable_scope('C'+scope):
        x = conv(x, 3, 1, 192)
    x = activation(x)
    with tf.variable_scope('D'+scope):
        x = conv(x, 3, 1, 192)
    x = activation(x)
    with tf.variable_scope('E'+scope):
        x = conv(x, 3, 1, 128)
    x = activation(x)
    x = _max_pool(x, ksize=3, stride=2)
    x = tf.reshape(x, [-1, 7*7*128])
    return x


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    #num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  trainable=True):
    var = tf.get_variable(name, shape, initializer=initializer)
    if weight_decay >0:
        wd = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('regularizer', wd)
    return var


def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
    biases = _get_variable('biases', shape=[filters_out], initializer=tf.zeros_initializer())
    return tf.nn.bias_add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME'), biases)
    #return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
