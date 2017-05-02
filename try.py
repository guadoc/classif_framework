import tensorflow as tf
import numpy as np
import math

def Cov(x):
    x = tf.Print(x, ["x", tf.shape(x)])
    shape = tf.shape(x)
    n_sample = shape[0]
    #n_sample = tf.Print(n_sample, [n_sample])
    num_units_in = shape[-1]
    #num_units_in = tf.Print(num_units_in, [num_units_in])
    means = tf.reshape(tf.reduce_mean(x, 0), [1, num_units_in])
    means = tf.Print(means, ["mean", tf.shape(means)])
    centered_x = tf.subtract(x, means)
    centered_x = tf.Print(centered_x, ["center", tf.shape(centered_x)])
    covs = tf.divide(tf.matmul(tf.matrix_transpose(centered_x), centered_x), tf.cast(n_sample - 1, tf.float32))
    #covs = tf.Print(covs, [covs])
    return covs


sess = tf.Session()
#
# sample = 3
# depth = 2
# input_size = 7
# filter_size = 2.
# n_filter = 1
# inputs = tf.get_variable('inputs', shape=[sample, input_size, input_size, depth], initializer=tf.truncated_normal_initializer( stddev=0.1))
# weights = tf.get_variable('weights', shape=[filter_size, filter_size, depth, n_filter], initializer=tf.truncated_normal_initializer( stddev=0.1))
# outputs = tf.nn.conv2d(inputs, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
# center = math.floor(input_size/2)
# decay = math.floor(filter_size/2)
# reg = 0
# for filter_i in range(n_filter):
#     out = outputs[:, center, center, filter_i]
#     inp = inputs[:, center-decay:center+decay+1 , center-decay:center+decay+1, :]
#     mask = tf.greater(out, tf.zeros_like(out))
#     positive_input = tf.boolean_mask(inp, mask)
#     if :
#         cov = Cov(positive_input)
#     else:
#         cov = tf.diag(tf.ones(depth))
#
#     vect = tf.reshape(weights[:,i], [1, weight_shape[-2]])
#     loss = tf.matmul(tf.matmul(vect, cov), tf.transpose(vect))
#     #reg += tf.log(loss[0, 0])
#     reg += loss[0, 0]


n_sample = 10
n_in_unit = 6
n_out_unit = 3
inputs = tf.get_variable('inputs', shape=[n_sample, n_in_unit], initializer=tf.truncated_normal_initializer( stddev=0.1))
weights = tf.get_variable('weights', shape=[n_in_unit, n_out_unit], initializer=tf.truncated_normal_initializer( stddev=0.1))
biases = tf.get_variable('biases', shape=[n_out_unit], initializer=tf.truncated_normal_initializer( stddev=0.001))
outputs = tf.nn.xw_plus_b(inputs, weights, biases)

reg = 0
for i in range(n_out_unit):
    mask = tf.greater(outputs[:, i], tf.zeros_like(outputs[:, i]))
    positive_input = tf.boolean_mask(inputs, mask)
    n_sample = tf.shape(positive_input)[0]
    cov = Cov(positive_input)
    vect = tf.reshape(weights[:,i], [n_in_unit, 1])
    #loss = tf.matmul(tf.matmul(vect, cov), tf.transpose(vect))
    loss = tf.matmul(tf.matmul(tf.transpose(vect), cov), vect)
    cond = tf.greater(n_sample, 1)
    val_0 = tf.constant(0.)
    reg+= tf.cond(cond, lambda: loss[0,0], lambda: val_0)


sess.run(tf.global_variables_initializer())


reg_ = sess.run(reg)
_ = sess.run(outputs)
print(reg_)
