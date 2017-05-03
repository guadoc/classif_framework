import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
import numpy as np
REG_COEF = 0.8
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01

CONV_WEIGHT_DECAY = 0.000000005
FC_WEIGHT_DECAY= 0.00000001


def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 201:
        lr = 0.01
    elif epoch < 300:
        lr = 0.001
    elif epoch < 400:
        lr = 0.0001
    elif epoch < 500:
        lr = 0.00001
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

def normal_density(x, mu, sigma2):
    centring_reduce = tf.divide(tf.square(tf.subtract(x, mu)), sigma2)
    regularizing_term = tf.sqrt(tf. multiply(sigma2, 2.*3.14))
    return tf.divide(tf.exp(centring_reduce ), regularizing_term )


def fc_me(x, num_units_out):
    collection = tf.get_variable_scope().name
    num_units_in = x.get_shape().as_list()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases  = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)

    moving_mean_1 = tf.get_variable('moving_mean_1', [num_units_out], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((num_units_out), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    #moving_std_1 = tf.get_variable('moving_std_1'+str(i), [num_units_out], initializer=tf.ones_initializer(), trainable=False)
    #p_mode_1 = tf.reshape(normal_density(outputs, moving_mean_1, moving_std_1), [1, -1])
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean( tf.multiply(tf.square(tf.subtract(outputs, mean_1)), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean( tf.multiply(tf.square(tf.subtract(outputs, mean_2)), p_mode_2) ,[0])  )
    reg = tf.add(var_1, var_2)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg, tf.shape(moving_mean_2), tf.shape(moving_mean_1)])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs


def fc_e(x, num_units_out):
    collection = tf.get_variable_scope().name
    num_units_in = x.get_shape().as_list()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases  = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    outputs = tf.nn.xw_plus_b(x, weights, biases)

    moving_mean_1 = tf.get_variable('moving_mean_1', [num_units_out], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((num_units_out), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    #moving_std_1 = tf.get_variable('moving_std_1'+str(i), [num_units_out], initializer=tf.ones_initializer(), trainable=False)
    #p_mode_1 = tf.reshape(normal_density(outputs, moving_mean_1, moving_std_1), [1, -1])
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply(tf.log( tf.square(tf.subtract(outputs, mean_1)) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply(tf.log( tf.square(tf.subtract(outputs, mean_2)) ), p_mode_2) ,[0])  )
    reg = tf.add(var_1, var_2)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg, tf.shape(moving_mean_2), tf.shape(moving_mean_1)])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, FC_WEIGHT_DECAY, name='reg'))
    return outputs


def conv_e(x, ksize, stride, filters_out):
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

    moving_mean_1 = tf.get_variable('moving_mean_1', [filters_out], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((filters_out), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply(tf.log( tf.square(tf.subtract(outs, mean_1)) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply(tf.log( tf.square(tf.subtract(outs, mean_2)) ), p_mode_2) ,[0])  )
    reg = tf.add(var_1, var_2)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg, tf.shape(moving_mean_2), tf.shape(moving_mean_1)])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
    return outputs


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

    moving_mean_1 = tf.get_variable('moving_mean_1', [filters_out], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((filters_out), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outs)
    p_mode_2 = tf.add(tf.multiply(p_mode_1, -1.), 1.)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(tf.multiply(p_mode_1, outs), [0]), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.reduce_mean(tf.multiply(p_mode_2, outs), [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean( tf.multiply(tf.square(tf.subtract(outs, mean_1)), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean( tf.multiply(tf.square(tf.subtract(outs, mean_2)), p_mode_2) ,[0])  )
    reg = tf.add(var_1, var_2)
    #reg = tf.Print(reg, [tf.get_variable_scope().name, reg, tf.shape(moving_mean_2), tf.shape(moving_mean_1)])
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, CONV_WEIGHT_DECAY, name='reg'))
    return outputs

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('layer_5'):
        n_out = 32
        x = conv_e(x, 5, 1, n_out)
        #x = tf.Print(x, ['1',tf.reduce_mean(x)])
        #tr_activation_summary = tf.summary.histogram('activation1', x[:,2,2,7], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)
    with tf.variable_scope('layer_4'):
        n_out = 64
        x = conv_e(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:,2,2,7], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)
    with tf.variable_scope('layer_3'):
        n_out = 64
        x = conv_e(x, 5, 1, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:,2,2,7], collections=['per_batch'])
        x = activation(x)
        x = max_pool_2x2(x)
        x = tf.reshape(x, [-1, 4*4*n_out])

    with tf.variable_scope('layer_2'):
        n_out = 1000
        x = fc_e(x, n_out)
        #x = tf.Print(x, ['2',tf.reduce_mean(x)])
        #tr_activation_summary = tf.summary.histogram('activation1', x[:,1], collections=['per_batch'])
        x = activation(x)
        #tr_activation_summary = tf.summary.histogram('activation2', x[:,1], collections=['per_batch'])
    with tf.variable_scope('layer_1'):
        outputs = fc_e(x, 10)
        #tr_activation_summary = tf.summary.histogram('activation1', outputs[:,1], collections=['per_batch'])
    return outputs
