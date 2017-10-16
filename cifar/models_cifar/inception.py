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

FC_WEIGHT_DECAY= 0.001
CONV_WEIGHT_DECAY = 0.001
regularization_conv = shade_conv#weight_decay
regularization_fc = shade#weight_decay

def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.07*math.pow(0.98, monitor.epoch-1)
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 20#12
    regs = []
    print('Layer number to regularize :'+ str(n_layers))
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def inception(x, f1_1, f3_3, layer, training_mode):
    with tf.variable_scope('conv1'):
        x1_1, params_conv1 = conv_2D(x, 1, 1, f1_1, conv_init, use_biases=False, padding='SAME')                
        #regularization_conv(x1_1, params_conv1, 'layer_'+str(layer), CONV_WEIGHT_DECAY)  
        x1_1, params_bn1 = bn(x1_1, training_mode)        
        regularization_conv(x1_1, [params_conv1, params_bn1], 'layer_'+str(layer), CONV_WEIGHT_DECAY)
                
    with tf.variable_scope('conv2'):
        x3_3, params_conv2 = conv_2D(x, 3, 1, f3_3, conv_init, use_biases=False, padding='SAME')                
        #regularization_conv(x3_3, params_conv2, 'layer_'+str(layer+1), CONV_WEIGHT_DECAY)  
        x3_3, params_bn2 = bn(x3_3, training_mode)        
        regularization_conv(x3_3, [params_conv2, params_bn2], 'layer_'+str(layer+1), CONV_WEIGHT_DECAY)
    return relu(tf.concat([x1_1, x3_3], -1)), [params_conv1, params_bn1, params_conv2, params_bn2]


def downsample(x, filter_conv, layer, training_mode):
    x_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv1'):
        x3_3, params_conv = conv_2D(x, 3, 2, filter_conv, conv_init, use_biases=False, padding='SAME')                
        #regularization_conv(x3_3, params_conv, 'layer_'+str(layer), CONV_WEIGHT_DECAY)  
        x3_3, params_bn = bn(x3_3, training_mode)        
        regularization_conv(x3_3, [params_conv, params_bn], 'layer_'+str(layer), CONV_WEIGHT_DECAY)  
        x3_3 = relu(x3_3)
    return tf.concat([x_pool, x3_3], -1), [params_conv, params_bn]



def inference(inputs, training_mode):
    x = inputs
    layer = 0

    layer=1
    with tf.variable_scope('layer_'+str(layer)):
        n_out = 96
        with tf.variable_scope('conv1'):
            x, params_conv = conv_2D(x, 3, 1, n_out, conv_init, use_biases=False, padding='VALID')            
            x, params_bn = bn(x, training_mode)            
            regularization_conv(x, [params_conv, params_bn], 'layer_'+str(layer), CONV_WEIGHT_DECAY)        
            x = relu(x)
    layer+=1

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 32, 32, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2
    
    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 32, 48, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

    with tf.variable_scope('layer_'+str(layer)):
        x, params = downsample(x, 80, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=1

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 112, 48, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 96, 64, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 80, 80, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 48, 96, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

    with tf.variable_scope('layer_'+str(layer)):
        x, params = downsample(x, 96, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=1

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 176, 160, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2
        
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.95), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_'+str(layer)):
        x, params = inception(x, 176, 160, layer, training_mode)
        #regularization_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=2

        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, [-1, 336])
        #x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.95), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_'+str(layer)):
        with tf.variable_scope('fc1'):
            outputs, params = fc(x, 10, fc_init)
        regularization_fc(outputs, params, 'layer_'+str(layer), FC_WEIGHT_DECAY)    

    return outputs, 0
