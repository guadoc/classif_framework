import numpy as np
import tensorflow as tf

class Data:
    def __init__(self, meta_data, session):
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.batch_sample = meta_data.build_input(self.batch_size)
        self.sess = session
        self.ndata = meta_data.image_number


    def sample(self, quantity):
        return self.sess.run(self.batch_sample, feed_dict={self.batch_size:quantity})

    def get_exemple(self):
        pass

    def get_next_batch(self, quantity):
        return None

    def get_batch(self, index):
        pass
