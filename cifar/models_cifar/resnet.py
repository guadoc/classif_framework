
import tensorflow as tf
import math
#from resnet import softmax_layer, conv_layer, residual_block
CONV_WEIGHT_DECAY = 0.00001

n_dict = {20:1, 32:2, 44:3, 56:4}

def activation(x):
    return tf.nn.relu(x)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.01*math.pow(0.95, monitor.epoch-1)
    print("lr: "+str(lr))
    return {"lr":lr, "momentum":momentum}


def layer_regularizer():
    n_layers = 1
    regs = []
    for i in range(n_layers):
        regs.append([tf.get_collection('layer_'+str(i+1)+'_reg'), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.005)
    weights = tf.Variable(initial, name=name)
    collection = tf.get_variable_scope().name
    tf.add_to_collection(collection+"_variables", weights)
    reg = tf.multiply(tf.nn.l2_loss(weights), CONV_WEIGHT_DECAY)
    tf.add_to_collection(collection+'_reg', reg)
    return weights

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))
    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)
    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    batch_norm = tf.nn.batch_norm_with_global_normalization( conv, mean, var, beta, gamma, 0.1, scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)
    return tf.nn.relu(conv)


def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt
    res = conv2 + input_layer
    return res
# ResNet architectures used for CIFAR-10
def inference(inputs, training_mode):
    n = 92
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    print(num_conv)
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inputs, [3, 3, 3, 16], 1)
        layers.append(conv1)

    for i in range (num_conv):
        with tf.variable_scope('layer_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [28, 28, 16]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [14, 14, 32]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [7, 7, 64]


    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]
        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)
    infos = out

    return layers[-1], infos
