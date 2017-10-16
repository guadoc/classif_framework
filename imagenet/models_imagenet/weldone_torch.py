import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block_weldone
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math





CONV_WEIGHT_STDDEV = 0.02
conv_init = tf.truncated_normal_initializer(stddev=0.1)


def optim_param_schedule_(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 5:
        lr = 0.00001
    elif epoch < 10:
        lr = 0.000001
    else:
        lr = 0.0000001    
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    if epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.001
    elif epoch < 50:
        lr = 0.0001
    elif epoch < 50:
        lr = 0.00001
    else:
        lr = 0.000001    
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}




CONV_WEIGHT_DECAY = 0.001

def layer_regularizer():
    n_layers = 35
    regs = []
    for i in range(n_layers):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs



def block(x, layer, n_block, strides, f_out, activation, training_mode, regul):
    for i in range(n_block):
        layer += 1
        stride = 1
        if i == 0: stride=strides
        ksizes = [1,3,1]
        strides = [1, stride, 1]
        filters_out = [f_out, f_out, 4*f_out]        
        with tf.variable_scope('res'+str(i+1)):
            x, params = residual_block_weldone(x, ksizes, strides, filters_out, conv_init, activation, training_mode)
            regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
            x = activation(x)
            #regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
    return x, layer


def inference(inputs, training_mode):
    regul = shade_conv#weight_decay
    layer = 0
    x = inputs
    #x = tf.Print(x , [ tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
    N_LAYER=1    
    with tf.variable_scope('scale_'+str(N_LAYER)):
        layer +=1
        with tf.variable_scope('conv1'):
            n_out = 64
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")            
            x, params = conv_2D(x, 7, 2, n_out, conv_init)                            
            x, params_bn = bn(x, training_mode)
            regul(x, [params, params_bn], 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x = relu(tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'))

    N_LAYER=2
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 64
        stride = 1
        x, layer = block(x, layer, 3, stride, f_out, relu, training_mode, regul)        

    N_LAYER=3
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 128
        stride = 2        
        x, layer = block(x, layer, 4, stride, f_out, relu, training_mode, regul)        

    N_LAYER=4
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 256
        stride = 2
        x, layer = block(x, layer, 23, stride, f_out, relu, training_mode, regul)
        

    N_LAYER=5
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 512
        stride = 2
        x, layer = block(x, layer, 3, stride, f_out, relu, training_mode, regul)        
    
    N_LAYER=6
    #x = tf.Print(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
    with tf.variable_scope('scale_'+str(N_LAYER)):
        layer += 1
        with tf.variable_scope('conv1'):    
            f_out = 1000
            stride = 1
            x, params = conv_2D(x, 1, 1, f_out, conv_init, use_biases=True)
            regul(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)

    
    # outputs = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    x = tf.transpose(tf.reshape(x, [- 1, 14*14, 1000]), [0, 2, 1])
    sorted_val, ind = tf.nn.top_k(  x,    k=196,    sorted=True)
    outputs = (tf.reduce_sum(sorted_val[:, :, 0:50], axis=2) + tf.reduce_sum(sorted_val[:, :, 146:], axis=2))/50

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
    