import tensorflow as tf
from layers.normalization import bn
import numpy as np


def fc(x, num_units_out, initializer):
    num_units_in = x.get_shape()[1]
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=initializer, dtype='float')
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    return outputs, [weights, biases]


def conv_2D(x, ksize, stride, filters_out, initializer, use_biases = False, padding = 'VALID'):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    weights =tf.get_variable('weights', shape=shape, initializer=initializer)    
    outputs = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
    params = [weights]
    if use_biases:
        biases = tf.get_variable('biases', shape=[filters_out], initializer=tf.zeros_initializer())
        outputs = tf.nn.bias_add(outputs, biases)
        params = [weights, biases]
    return outputs, params




def residual_block_nb(x, ksizes, strides, filters_out, initializer, activation, is_training):
    filters_in = x.get_shape()[-1]
    tot_stride = 0
    params = []
    shortcut = x
    for i in range(len(filters_out)-1):
        with tf.variable_scope('conv'+str(i+1)): 
            x, w = conv_2D(x, ksizes[i], strides[i], filters_out[i], initializer, use_biases=False, padding='SAME')
            params+= w
            tot_stride += strides[i]
            x, param_bn = bn(x, is_training)
            x = activation(x)

    with tf.variable_scope('conv'+str(len(filters_out))):             
        x, w = conv_2D(x, ksizes[-1], strides[-1], filters_out[-1], initializer, use_biases=False, padding='SAME')
        params += w
        tot_stride += strides[-1]
        x, param_bn = bn(x, is_training)
        
    if filters_out[-1] != filters_in or tot_stride > len(filters_out):
        with tf.variable_scope('shortcut'):
            shortcut, w = conv_2D(shortcut, 1, tot_stride - len(filters_out)+1, filters_out[-1], initializer, use_biases=False, padding='SAME')
            params+= w
    with tf.variable_scope('shortcut'):
        shortcut, param_bn = bn(shortcut, is_training) 
    return activation(x + shortcut), params




def residual_block_weldone(x, ksizes, strides, filters_out, initializer, activation, is_training):
    filters_in = x.get_shape()[-1]
    tot_stride = 0
    params = []
    shortcut = x       
    with tf.variable_scope('conv1'):
        tot_stride += strides[0]
        x, w = conv_2D(x, ksizes[0], strides[0], filters_out[0], initializer)                
        x, params_bn = bn(x, is_training)
        params+= w
        params+= params_bn
        x = activation(x)    
    with tf.variable_scope('conv2'):        
        tot_stride += strides[1]                
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")        
        x, w = conv_2D(x, ksizes[1], strides[1], filters_out[1], initializer)                
        x, params_bn = bn(x, is_training)
        params+= w
        params+= params_bn
        x = activation(x)    
    with tf.variable_scope('conv3'):          
        tot_stride += strides[2]
        x, w = conv_2D(x, ksizes[2], strides[2], filters_out[2], initializer)
        x, params_bn = bn(x, is_training)
        params+= w
        params+= params_bn

    if filters_out[-1] != filters_in or tot_stride > len(filters_out):        
        with tf.variable_scope('shortcut'):
            shortcut, w = conv_2D(shortcut, 1, tot_stride - len(filters_out)+1, filters_out[-1], initializer)
            shortcut, params_bn = bn(shortcut, is_training)
            params+= w
            params+= params_bn
    #return activation(x + shortcut), params
    return x + shortcut, params



def shake_block(x, ksizes, strides, filters_out, initializer, activation, is_training):
    shape = x.get_shape()
    filters_in = shape[-1]
    params = []
    shortcut = activation(x)
    if strides[0] == 2:
        sh1 = tf.nn.avg_pool(x, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        sh1, w = conv_2D(sh1, 1, 1, filters_in, initializer, 'convshortcut1' , False, 'VALID')
        params+= w
        sh2 = tf.image.crop_to_bounding_box(x, 0, 0, shape[1]-1, shape[2]-1)
        sh2 = tf.image.pad_to_bounding_box(sh2, 1, 1, shape[1], shape[2])
        sh2 = tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        sh2, w = conv_2D(sh2, 1, 1, filters_in, initializer, 'convshortcut2' , False, 'VALID')
        params+= w
        shortcut = tf.concat([sh1, sh2], axis=-1)
    for branch in range(2):
        x_ = x
        for i in range(len(filters_out)-1):
            x_ = activation(x_)
            x_, w = conv_2D(x_, ksizes[i], strides[i], filters_out[i], initializer, 'conv' +  str(branch*(i+1)), False, padding='SAME')
            params+= w
            x_ = bn(x_, is_training, 'bn' + str(branch*(i+1)))
        shortcut += x_
    return shortcut, params
