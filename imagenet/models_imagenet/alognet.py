
import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
REG_COEF = 0.6
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
CONV_WEIGHT_DECAY = 0.000005
FC_WEIGHT_DECAY= 0.00002


def optim_param_schedule(epoch):
    momentum = 0.9
    lr = 0.01
    return {"lr":lr, "momentum":momentum}


def regularizer():
    return tf.get_collection('reg')


def e_regularizer(x, weights):
    weight_shape = weights.get_shape().as_list()#tf.shape(filter)
    _, variance = tf.nn.moments(x, axes=[0])
    moving_std = tf.get_variable('moving_variance', variance.get_shape(), initializer=tf.ones_initializer(), trainable=False)
    update_moving_std = moving_averages.assign_moving_average(moving_std, tf.sqrt(variance), REG_COEF)
    reg=0
    for i in range(weight_shape[-1]):
        vect = weights[:,i]
        vect = tf.Print(vect, [tf.shape(vect)])
        reg += tf.log(tf.nn.l2_loss(tf.multiply(update_moving_std, vect)))
    return tf.multiply(reg, FC_WEIGHT_DECAY, name='weight_loss')

def e_cross_regularizer(x, weights):
    weight_shape = weights.get_shape().as_list()
    shape = tf.shape(x)
    mean = tf.reshape(tf.reduce_mean(x, 0), [1, shape[-1]])
    cov = tf.subtract(tf.divide(     tf.matmul(tf.matrix_transpose(x), x),         tf.cast(shape[0]-1, tf.float32)) ,
          tf.multiply(   tf.matmul(tf.matrix_transpose(mean), mean),   tf.cast(shape[0], tf.float32) /tf.cast(shape[0]-1, tf.float32)  ) )

    moving_cov = tf.get_variable('moving_variance', cov.get_shape(), initializer=tf.ones_initializer(), trainable=False)
    updated_moving_cov = moving_averages.assign_moving_average(moving_cov, cov, REG_COEF)
    reg=0
    for i in range(weight_shape[-1]):
        vect = tf.reshape(weights[:,i], [1, weight_shape[-2]])
        loss = tf.matmul(tf.matmul(vect, updated_moving_cov), tf.transpose(vect))
        reg+= tf.log(loss[0, 0])
    #reg = tf.Print(reg, [reg])
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
        reg += tf.log(tf.nn.l2_loss(tf.multiply(update_moving_std, weights[:,:,:,i])))
    return tf.multiply(reg, CONV_WEIGHT_DECAY, name='weight_loss')


def fc(x, num_units_out):
    num_units_in = x.get_shape()[-1]
    weights_initializer = tf.truncated_normal_initializer( stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    reg = e_regularizer(x, weights)
    tf.add_to_collection('reg', reg)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights =tf.get_variable('weights', shape=shape, initializer=initializer)
    truc = weights.get_shape()
    reg = e_regularizer_filter(x, weights)
    tf.add_to_collection('reg', reg)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

activation = tf.nn.relu



def inference(inputs, training_mode):
    num_classes = 1000
    mul=2
    is_training = training_mode
    x1 = convs(inputs, '1')
    x2 = convs(inputs, '2')

    x = tf.concat([x1,x2], 1)
    x = activation(x)
    x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.5), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('fc1'):
        x = fc(x, 2048*mul)
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
