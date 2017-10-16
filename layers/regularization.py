
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


def weight_decay(outputs, params, collection, beta):
    reg = 0
    for w in params:
        reg += tf.nn.l2_loss(w)
        tf.add_to_collection(collection+"_variables", w)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, beta, name='reg'))

def shade(outputs, params, collection, beta):
    REG_COEF = 0.8
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.ones_initializer(), trainable=False)
    moving_mean_2 = tf.Variable(np.full((n_units), -1., dtype=np.float32), name='moving_mean_2', trainable=False)
    p_mode_1 = tf.sigmoid(outputs)
    p_mode_2 = 1 - p_mode_1#tf.add(tf.multiply(p_mode_1, -1.), 1.)
    p_a1 = tf.reduce_mean(p_mode_1, [0])
    p_a2 = 1 - p_a1#tf.reduce_mean(p_mode_2, [0])
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_a1), REG_COEF, zero_debias=False)
    mean_2 = moving_averages.assign_moving_average(moving_mean_2, tf.divide(tf.reduce_mean(tf.multiply(p_mode_2, outputs), [0]), p_a2), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_1) ), p_mode_1) ,[0])  )
    var_2 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.square(tf.subtract(outputs, mean_2) ), p_mode_2) ,[0])  )
    for w in params:
        #reg += tf.nn.l2_loss(w)
        tf.add_to_collection(collection+"_variables", w)
    reg = var_1 + var_2 
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, beta, name='reg'))




def shade_conv(outputs, params, collection, beta):
    shape_output = outputs.get_shape()
    sizeX = shape_output[1]
    sizeY = shape_output[2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    outs = outputs[:, indexX, indexY, :]
    shade(outs, params, collection, beta)


def var_entropy(outputs, params, collection, beta):
    REG_COEF = 0.8
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean', [n_units], initializer=tf.zeros_initializer(), trainable=False)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(outputs, [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(tf.reduce_mean( tf.square( tf.subtract(outputs, mean_1) ) ,[0])  )
    for weights in params:
        tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, beta, name='reg'))
