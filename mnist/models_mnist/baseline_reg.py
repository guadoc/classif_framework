import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
import numpy as np
REG_COEF = 0.8
FC_WEIGHT_STDDEV=0.01
FC_WEIGHT_DECAY = 0.0005



def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    print("FC_WEIGHT_DECAY: "+str(FC_WEIGHT_DECAY))
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 50:
        lr = 0.05
    elif epoch < 100:
        lr = 0.01
    elif epoch < 200:
        lr = 0.001
    else:
        lr = 0.00005
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 3
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def regul(outputs):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_1) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_2) ), p_mode_2) ,[0])  )
    return tf.add(var_1, var_2)


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    collection = tf.get_variable_scope().name
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, dtype='float')
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    reg = regul(outputs)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs

def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_3'):
        n_out = 1024
        x = fc(x, n_out)
        x = activation(x)
    with tf.variable_scope('layer_2'):
        n_out = 1024
        x = fc(x, n_out)
        x = activation(x)
    with tf.variable_scope('layer_1'):
        outputs = fc(x, 10)

    return outputs, outputs
