import tensorflow as tf

def max_pool_2D(x, size, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                        strides=[1, stride, stride, 1], padding=padding)
