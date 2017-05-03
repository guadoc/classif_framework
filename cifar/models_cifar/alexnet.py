import tensorflow as tf
import math
REG_COEF = 0.9
FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
CONV_WEIGHT_DECAY = 0#0.0005
FC_WEIGHT_DECAY= 0#0.0005


#activation = reluu#tf.nn.relu

def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.01*math.pow(0.95, monitor.epoch-1)
    return {"lr":lr, "momentum":momentum}

def regularizer():
    return tf.get_collection('reg')

def layer_regularizer():
    n_layers = 2
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    collection = tf.get_variable_scope().name
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, dtype='float')
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
    tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+"_variables", biases)
    reg = tf.multiply(tf.nn.l2_loss(weights), CONV_WEIGHT_DECAY)
    tf.add_to_collection(collection+'_reg', reg)
    return tf.nn.xw_plus_b(x, weights, biases)

def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    collection = tf.get_variable_scope().name
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    #weights = _get_variable('weights', collection = tf.get_variable_scope().name, shape=shape, initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
    weights = tf.get_variable('weights',
                         shape=shape,
                         initializer=initializer,
                         dtype='float')
    tf.add_to_collection(collection+"_variables", weights)
    reg = tf.multiply(tf.nn.l2_loss(weights), CONV_WEIGHT_DECAY)
    tf.add_to_collection(collection+'_reg', reg)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')





def inference(inputs, training_mode):
    x = inputs

    with tf.variable_scope('layer_5'):
        n_out = 128
        x = conv(x, 5, 2, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = activation(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.lrn(x)

    with tf.variable_scope('layer_4'):
        n_out = 128
        x = conv(x, 5, 2, n_out)
        #tr_activation_summary = tf.summary.histogram('activation1', x[:, 2, 2, 0], collections=['per_batch'])
        x = activation(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.lrn(x)

        #x = tf.Print(x, [tf.shape(x)])
        x = tf.reshape(x, [-1, 2*2*n_out])

    with tf.variable_scope('layer_3'):
        n_out = 384
        x = fc(x, n_out)
        x = activation(x)

    with tf.variable_scope('layer_2'):
        n_out = 192
        x = fc(x, n_out)
        x = activation(x)

    with tf.variable_scope('layer_1'):
        outputs = fc(x, 10)
        #tr_activation_summary = tf.summary.histogram('activation1', outputs[:, 2], collections=['per_batch'])

    return outputs
