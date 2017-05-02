
import tensorflow as tf
#from resnet import softmax_layer, conv_layer, residual_block
CONV_WEIGHT_DECAY = 0.00001

n_dict = {20:1, 32:2, 44:3, 56:4}


def optim_param_schedule(epoch):
    momentum = 0.9
    lr = 0.0001
    return {"lr":lr, "momentum":momentum}


def regularizer():
    return tf.get_collection('reg')


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

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
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)
    return tf.nn.relu(conv)


def conv_layer_(inpt, filter_shape, stride):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    out = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    reg = 0
    reg = tf.multiply(reg, CONV_WEIGHT_DECAY, name='weight_loss')
    #x = tf.Print(x, [reg])
    tf.add_to_collection('reg', reg)
    return out


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
    res = tf.log( conv2 + 0.03) + input_layer
    return res
# ResNet architectures used for CIFAR-10
def inference(inputs, training_mode):
    n = 20
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inputs, [3, 3, 3, 16], 1)
        layers.append(conv1)
        tr_activation_summary = tf.summary.histogram('activation1', conv1[:, 4, 4, 0], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation2', conv1[:, 4, 4, 7], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation3', conv1[:, 4, 4, 12], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation4', conv1[:, 4, 4, 3], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation5', conv1[:, 4, 4, 8], collections=['per_batch'])
        tr_activation_summary = tf.summary.histogram('activation6', conv1[:, 4, 4, 14], collections=['per_batch'])

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)
            tr_activation_summary = tf.summary.histogram('activation1', conv4[:, 4, 4, 0], collections=['per_batch'])
            tr_activation_summary = tf.summary.histogram('activation2', conv4[:, 4, 4, 7], collections=['per_batch'])
            tr_activation_summary = tf.summary.histogram('activation3', conv4[:, 4, 4, 12], collections=['per_batch'])
            tr_activation_summary = tf.summary.histogram('activation4', conv4[:, 4, 4, 3], collections=['per_batch'])
            tr_activation_summary = tf.summary.histogram('activation5', conv4[:, 4, 4, 34], collections=['per_batch'])
            tr_activation_summary = tf.summary.histogram('activation6', conv4[:, 4, 4, 63], collections=['per_batch'])

        assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]
        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)

    return layers[-1]
