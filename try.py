import tensorflow as tf
#import tensorflow.contrib as contrib
import numpy as np
import math
sess = tf.Session()



dist = tf.contrib.distributions.Normal(mu=[[2., 1.], [2., 1.]], sigma=[[1., 4], [1., 4]] )
x = tf.ones([5,2, 2])

a = dist.cdf(x)
a = tf.Print(a, [tf.shape(a)])



sess.run(tf.global_variables_initializer())


a_ = sess.run([a])

print(a_)
