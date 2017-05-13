import tensorflow as tf
from tensorflow.python.training import moving_averages
#import math
import numpy as np
REG_COEF = 0.8
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01

CONV_WEIGHT_DECAY = 0.0001
FC_WEIGHT_DECAY= 0.0005
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
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def regularizer():
    return tf.get_collection('reg')

def activation(x):
    return tf.nn.relu(x)

def proba_mode(x, mu_1, std_1, mu_2, std_2, p_1, p_2):
    dist_1 = tf.contrib.distributions.Normal(mu=mu_1, sigma=std_1)
    dist_2 = tf.contrib.distributions.Normal(mu=mu_2, sigma=std_2)
    p_x_1 = dist_1.cdf(x)
    p_x_2 = dist_2.cdf(x)
    p_a1 = tf.multiply(p_x_1, p_1)
    tot = p_a1 + tf.multiply(p_x_2, p_2)
    #tot = tf.Print(tot, [name,"tot",  tf.reduce_min(tot), tf.reduce_min(p_x_1),  tf.reduce_min(p_x_2), tf.reduce_min(x), tf.reduce_max(x)])
    proba = tf.divide(p_a1, tot + 0.000000001)
    return p_x_1, p_x_2, proba, tf.add(tf.multiply(proba, -1.), 1.)


def regul(outputs):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    moving_std_1 = tf.get_variable('moving_std_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_std_2 = tf.get_variable('moving_std_2', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_p_1 = tf.Variable(np.full((n_units), 0.5, dtype=np.float32), name='p_1', trainable=False)
    moving_p_2 = tf.Variable(np.full((n_units), 0.5, dtype=np.float32), name='p_2', trainable=False)
    p_y_1, p_y_2, p_mode_1, p_mode_2 = proba_mode(outputs, moving_mean_1, moving_std_1, moving_mean_2, moving_std_2, moving_p_1, moving_p_2)
    n_1 = tf.reduce_sum(p_mode_1, [0])[0]
    n_2 = tf.reduce_sum(p_mode_2, [0])[0]
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_y_1, outputs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_y_2, outputs), [0]), REG_COEF, zero_debias=False)
    ind_1 = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(outputs, mean_1)), p_y_1), [0])
    ind_2 = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(outputs, mean_2)), p_y_2), [0])
    std_1 = moving_averages.assign_moving_average(moving_std_1, ind_1, REG_COEF, zero_debias=False)
    std_2 = moving_averages.assign_moving_average(moving_std_2, ind_2, REG_COEF, zero_debias=False)
    p_1 = moving_averages.assign_moving_average(moving_p_1, tf.reduce_mean(p_y_1, [0]), REG_COEF, zero_debias=False)
    p_2 = moving_averages.assign_moving_average(moving_p_2, tf.reduce_mean(p_y_2, [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum( ind_1 )
    var_2 = tf.reduce_sum( ind_2 )
    with tf.control_dependencies([p_2, p_1, std_1, std_2]):
        result = tf.add(var_1, var_2)
    return result

def regul_(outputs):
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    p_a1 = tf.reduce_mean(p_mode_1, [0])
    p_a2 = tf.reduce_mean(p_mode_2, [0])
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_a1), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.divide(tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), p_a2), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_1) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_2) ), p_mode_2) ,[0])  )
    return tf.add(var_1, var_2)

def conv_me(x, ksize, stride, filters_out):
    collection = tf.get_variable_scope().name
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights =tf.get_variable('weights', shape=shape, initializer=initializer)
    tf.add_to_collection(collection+"_variables", weights)
    outputs = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    shape_output = outputs.get_shape()
    sizeX = shape_output[1]
    sizeY = shape_output[2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    outs = outputs[:, indexX, indexY, :]
    reg = regul_(outs)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
    return outputs


def fc_me(x, num_units_out):
    collection = tf.get_variable_scope().name
    num_units_in = x.get_shape().as_list()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases  = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    reg = regul_(outputs)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_5'):
        n_out = 32
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)

    with tf.variable_scope('layer_4'):
        n_out = 64
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)

    with tf.variable_scope('layer_3'):
        n_out = 64
        x = conv_me(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)
        infos = x
        x = tf.reshape(x, [-1, 4*4*n_out])

    with tf.variable_scope('layer_2'):
        n_out = 1000
        x = fc_me(x, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2], collections=['per_batch'])
        x = activation(x)
    with tf.variable_scope('layer_1'):
        outputs = fc_me(x, 10)
        #tr_activation_summary = tf.summary.histogram('activation1', outputs[:, 2], collections=['per_batch'])

    return outputs, infos
