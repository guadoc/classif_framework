import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
#from layers.trainable import fc, conv_2D#, residual_block_weldone
#from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops



CONV_WEIGHT_DECAY = 0#0.005


CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.truncated_normal_initializer(stddev=0.1)
fc_init = tf.random_normal_initializer(0.01)#tf.uniform_unit_scaling_initializer(factor=1.0)


def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 7:
        lr = 0.00001
    elif epoch < 15:
        lr = 0.000001
    else:
        lr = 0.0000001    
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers =35
    regs = []
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs


def conv_2D(x, ksize, stride, filters_out, initializer, use_biases = False, padding = 'VALID', field=''):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    weights =tf.get_variable(field+'/weights', shape=shape, initializer=initializer)    
    outputs = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
    params = [weights]
    if use_biases:
        biases = tf.get_variable(field+'/biases', shape=[filters_out], initializer=tf.zeros_initializer())
        outputs = tf.nn.bias_add(outputs, biases)
        params = [weights, biases]
    return outputs, params



def bn(x, is_training, field=''):
    BN_DECAY = 0.9#0.9997
    BN_EPSILON = 0.00001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = tf.get_variable(field+'/beta', params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(field+'/gamma', params_shape, initializer=tf.ones_initializer())
    moving_mean     = tf.get_variable(field+'/moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf.get_variable(field+'/moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)
    tf.add_to_collection("variable_to_save", moving_mean)
    tf.add_to_collection("variable_to_save", moving_variance)
    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    update_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY), lambda: moving_mean)
    update_moving_variance = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY), lambda: moving_variance)

    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)




def residual_block_weldone(x, ksizes, strides, filters_out, initializer, activation, is_training, l):
    filters_in = x.get_shape()[-1]
    tot_stride = 0
    params = []
    shortcut = x       
    #with tf.variable_scope(l+'/conv1'):
    field = l + '/conv1'
    x, w = conv_2D(x, ksizes[0], strides[0], filters_out[0], initializer, field= field)
    params+= w
    tot_stride += strides[0]
    x = bn(x, is_training=is_training, field= field)
    x = activation(x)    
    
    field = l + '/conv2'
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")        
    x, w = conv_2D(x, ksizes[1], strides[1], filters_out[1], initializer, field= field)
    params+= w
    tot_stride += strides[1]
    x = bn(x, is_training=is_training, field= field)
    x = activation(x)    
    field = l + '/conv3'
    x, w = conv_2D(x, ksizes[2], strides[2], filters_out[2], initializer, field= field)
    params+= w
    tot_stride += strides[2]
    x = bn(x, is_training=is_training, field= field)

    if filters_out[-1] != filters_in or tot_stride > len(filters_out):        
        field = l + '/shortcut'
        shortcut, w = conv_2D(shortcut, 1, tot_stride - len(filters_out)+1, filters_out[-1], initializer, field= field)        
        shortcut = bn(shortcut, is_training=is_training, field= field)
        params+= w
    return activation(x + shortcut), params



def block(x, layer, n_block, strides, f_out, activation, training_mode, regul):
    l = layer
    for i in range(n_block):
        layer += 1
        stride = 1
        if i == 0: stride=strides
        ksizes = [1,3,1]
        strides = [1, stride, 1]
        filters_out = [f_out, f_out, 4*f_out]  
        x, params = residual_block_weldone(x, ksizes, strides, filters_out, conv_init, activation, training_mode, str(l)+'/res'+str(i+1))
        regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
    return x, layer


def inference(inputs, training_mode):
    regul = weight_decay#shade_conv
    layer = 0
    x = inputs
    #x = tf.Print(x , [ tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
    N_LAYER=1    
    layer +=1   
    #with tf.variable_scope('conv1'):
    n_out = 64
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")            
    x, params = conv_2D(x, 7, 2, n_out, conv_init, field='ok')                    
    x = bn(x, training_mode, field='ok')
    regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
    x = relu(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    scale=2    
    f_out = 64
    stride = 1
    with tf.variable_scope('layer_1'):
        x, layer = block(x, layer, 3, stride, f_out, relu, training_mode, regul)        
    #x = tf.Print(x, [tf.reduce_mean(x)])

    scale=3
    f_out = 128
    stride = 2     
    with tf.variable_scope('layer_2'):   
        x, layer = block(x, layer, 4, stride, f_out, relu, training_mode, regul)        

    scale=4    
    f_out = 256
    stride = 2
    
    with tf.variable_scope('layer_3'):   
        x, layer = block(x, layer, 23, stride, f_out, relu, training_mode, regul)        

    scale=5    
    f_out = 512
    stride = 2

    with tf.variable_scope('layer_4'):   
        x, layer = block(x, layer, 3, stride, f_out, relu, training_mode, regul)        
    
    scale=6        
    layer += 1
    with tf.variable_scope('last1'):    
        f_out = 1000
        stride = 1
        x, params = conv_2D(x, 1, 1, f_out, conv_init, use_biases=True, field='pk')
        regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)    
        #outputs = tf.reduce_mean(x, reduction_indices=[1, 2], name="MERDE")

    
    x = tf.reshape(x, [- 1, 14*14, 1000])    
    x = tf.transpose(x, [0, 2, 1])
    sorted_val, ind = tf.nn.top_k(  x,    k=196,    sorted=True,    name=None)
    outputs = (tf.reduce_sum(sorted_val[:, :, 0:50], axis=2) + tf.reduce_sum(sorted_val[:, :, 146:], axis=2))/50

    # outputs = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    
    print('ResNet with '+str(layer) + ' layer and '+str(N_LAYER)+' sclale')
    return outputs, 0




    # x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    # N_LAYER+=1
    # with tf.variable_scope('scale_'+str(N_LAYER)):
    #     with tf.variable_scope('conv1'):
    #         layer += 1
    #         n_out = 1000
    #         outputs, params = fc(x, n_out, conv_init)
    #     #regul(outputs, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
    # return outputs, outputs            
    