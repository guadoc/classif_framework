import tensorflow as tf
from layers.activation import relu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math

FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)


CONV_WEIGHT_DECAY = 10#005
FC_WEIGHT_DECAY= 1#001
#da
#CONV = 0.000005 < 0.00005
#FC = 0.000001 < 0.000001
#sans da
#CONV = 0.000005 __last test = 0.000001
#FC   = 0.000001 __last test = 0.000005
def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 120:
        lr = 0.01
    elif epoch < 180:
        lr = 0.001
    else:
        lr = 0.00001
    lr = 0.08*math.pow(0.98, epoch-1)
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 5
    regs = []
    print('Layer number to regularize: '+ str(n_layers))
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def inference(inputs, training_mode):
    regularization_conv = shade_conv#weight_decay
    regularization_fc = shade#weight_decay
    x = inputs
    with tf.variable_scope('layer_1'):
        n_out = 32
        x, params = conv_2D(x, 5, 1, n_out, conv_init, 'conv1', True)
        regularization_conv(x, params, tf.get_variable_scope().name, 'conv1', CONV_WEIGHT_DECAY)
        x = relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer_2'):
        n_out = 64
        x, params = conv_2D(x, 5, 1, n_out, conv_init, 'conv1', True)
        regularization_conv(x, params, tf.get_variable_scope().name, 'conv1', CONV_WEIGHT_DECAY)
        x = relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer_3'):
        n_out = 64
        x, params = conv_2D(x, 5, 1, n_out, conv_init, 'conv1', True)
        regularization_conv(x, params, tf.get_variable_scope().name, 'conv1', CONV_WEIGHT_DECAY)
        x = relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.reshape(x, [-1, 4*4*n_out])
    #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.8), lambda: tf.nn.dropout(x, 1))
    with tf.variable_scope('layer_4'):
        n_out = 1000
        x, params = fc(x, n_out, fc_init, 'fc1')
        regularization_fc(x, params, tf.get_variable_scope().name, 'fc1', FC_WEIGHT_DECAY)
        x = relu(x)

    #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.7), lambda: tf.nn.dropout(x, 1))
    with tf.variable_scope('layer_5'):
        n_outputs = 10
        outputs, params = fc(x, n_outputs, fc_init, 'fc1')
        regularization_fc(outputs, params, tf.get_variable_scope().name, 'fc1', FC_WEIGHT_DECAY)

    return outputs, tf.nn.softmax(outputs)
