import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
REG_COEF = 0.6
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01

CONV_WEIGHT_DECAY = 0.
FC_WEIGHT_DECAY= 0.


def optim_param_schedule(epoch):
    momentum = 0.9
    lr = 0.01
    return {"lr":lr, "momentum":momentum}


def regularizer():
    return tf.get_collection('reg')


def activation(x):
    return tf.nn.relu(x)


def Cov(x):
#x = tf.Print(x, ["x", tf.shape(x)])
    shape = tf.shape(x)
    n_sample = shape[0]
    #n_sample = tf.Print(n_sample, [n_sample])
    num_units_in = shape[-1]
    #num_units_in = tf.Print(num_units_in, [num_units_in])
    means = tf.reshape(tf.reduce_mean(x, 0), [1, num_units_in])
    means = tf.Print(means, ["mean", tf.shape(means)])
    centered_x = tf.subtract(x, means)
    centered_x = tf.Print(centered_x, ["center", tf.shape(centered_x)])
    covs = tf.divide(tf.matmul(tf.matrix_transpose(centered_x), centered_x), tf.cast(n_sample - 1, tf.float32))
    #covs = tf.Print(covs, [covs])
    return covs


def fc_e(x, num_units_out):
    num_units_in = x.get_shape().as_list()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    outputs = tf.nn.xw_plus_b(x, weights, biases)
    #outputs = tf.Print(outputs, ["o before", tf.reduce_mean(outputs)])
    reg = 0.
    for i in range(num_units_out):
        mask = tf.greater(outputs[:, i], tf.zeros_like(outputs[:, i]))
        positive_input = tf.boolean_mask(x, mask)
        n_sample_pos = tf.shape(positive_input)[0]
        cond = tf.greater(n_sample_pos, 1)
        positive_input = tf.Print(positive_input, ["x", tf.shape(positive_input)])
        cov = Cov(positive_input)#
        vect = tf.reshape(weights[:,i], [num_units_in, 1])
        loss = tf.matmul(tf.matmul(tf.transpose(vect), cov), vect)
        val_0 = tf.constant(0.)
        reg += tf.cond(cond, lambda: loss[0,0], lambda: val_0)
    reg = tf.multiply(reg, FC_WEIGHT_DECAY, name='weight_loss')
    #outputs = tf.Print(outputs, ["o after", tf.reduce_mean(outputs)])
    tf.add_to_collection('reg', reg)
    return outputs

def e_cross_regularizer(x, weights):
    weight_shape = weights.get_shape().as_list()
    shape = tf.shape(x)
    mean = tf.reshape(tf.reduce_mean(x, 0), [1, shape[-1]])
    cov = tf.subtract(tf.divide(     tf.matmul(tf.matrix_transpose(x), x),         tf.cast(shape[0]-1, tf.float32)) ,
          tf.multiply(   tf.matmul(tf.matrix_transpose(mean), mean),   tf.cast(shape[0], tf.float32) /tf.cast(shape[0]-1, tf.float32)  ) )
    moving_cov = tf.get_variable('moving_covariance', cov.get_shape(), initializer=tf.ones_initializer(), trainable=False)
    for i in range(weight_shape[-1]):
        vect = tf.reshape(weights[:,i], [1, weight_shape[-2]])
        loss = tf.matmul(tf.matmul(vect, updated_moving_cov), tf.transpose(vect))
        #reg += tf.log(loss[0, 0])
        reg += loss[0, 0]
    return tf.multiply(reg, FC_WEIGHT_DECAY, name='weight_loss')


def e_regularizer_filter(x, weights):
    weight_shape = weights.get_shape().as_list()
    input_shape = x.get_shape().as_list()
    index = math.floor(float(input_shape[1])*0.5) - math.floor(float(weight_shape[0])*0.5)
    vect_ = x[:, index:index+weight_shape[0], index:index+weight_shape[0],:]
    _, variance = tf.nn.moments(vect_, axes=[0])
    moving_std = tf.get_variable('moving_variance', variance.get_shape(), initializer=tf.ones_initializer(), trainable=False)
    update_moving_std = moving_averages.assign_moving_average(moving_std, tf.sqrt(variance), REG_COEF)
    reg=0
    for i in range(weight_shape[-1]):
        #reg += tf.log(tf.nn.l2_loss(tf.multiply(update_moving_std, weights[:,:,:,i])))
        reg += tf.nn.l2_loss(tf.multiply(update_moving_std, weights[:,:,:,i]))
    return tf.multiply(reg, CONV_WEIGHT_DECAY, name='weight_loss')


def fc(x, num_units_out):
    num_units_in = x.get_shape()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x



def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights =tf.get_variable('weights', shape=shape, initializer=initializer)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')





def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(inputs, training_mode):
    x = inputs
    with tf.variable_scope('scale1'):
        x = activation(conv(x, 5, 1, 32))
        x = max_pool_2x2(x)
    with tf.variable_scope('scale2'):
        x = activation(conv(x, 5, 1, 64))
        x = max_pool_2x2(x)
    with tf.variable_scope('scale3'):
        x = activation(conv(x, 5, 1, 64))
        x = max_pool_2x2(x)
    x = tf.reshape(x, [-1, 4*4*64])

    with tf.variable_scope('fc1'):
        x = activation(fc_e(x, 1000))
        tr_activation_summary = tf.summary.histogram('activation1', x[:,1], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation2', x[:,8], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation3', x[:,100], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation4', x[:,500], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation5', x[:,501], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation6', x[:,750], collections=['per_batch'])
    with tf.variable_scope('fc2'):
        outputs = fc(x, 10)
    return outputs
