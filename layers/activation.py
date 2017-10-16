import tensorflow as tf

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, alpha):
    return tf.maximum(alpha*x, x)
