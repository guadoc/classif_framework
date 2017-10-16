import tensorflow as tf
import math
from layers.trainable import fc
from layers.activation import relu as activation
from layers.regularization import weight_decay, shade as regularization, shade_conv


REG_COEF = 0.9

FC_WEIGHT_STDDEV=0.01
fc_init = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
FC_WEIGHT_DECAY= 0.#00005


def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.01*math.pow(0.97, epoch-1)
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(FC_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}



def layer_regularizer():
    n_layers = 1
    regs = []
    print('Layer number to regularize :'+ str(n_layers))
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def inference(inputs, training_mode):
    x = tf.reshape(inputs, [-1, 28*28*3])

    with tf.variable_scope('layer_4'):
        n_out = 512
        x, params = fc(x, n_out, fc_init, 'fc1')
        #regularization(x, params, tf.get_variable_scope().name, 'fc1', FC_WEIGHT_DECAY)
        x = activation(x)

    with tf.variable_scope('layer_3'):
        n_out = 512
        x, params = fc(x, n_out, fc_init, 'fc2')
        #regularization(x, params, tf.get_variable_scope().name, 'fc2', FC_WEIGHT_DECAY)
        x = activation(x)
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.90), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_2'):
        n_out = 512
        x, params = fc(x, n_out, fc_init, 'fc3')
        #regularization(x, params, tf.get_variable_scope().name, 'fc3', FC_WEIGHT_DECAY)
        x = activation(x)
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.90), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_1'):
        outputs, params = fc(x, 10, fc_init, 'fc4')
        regularization(outputs, params, tf.get_variable_scope().name, 'fc4', FC_WEIGHT_DECAY)

    return outputs, tf.nn.softmax(outputs)
