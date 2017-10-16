import numpy as np
import tensorflow as tf

class Data:
    def __init__(self, meta_data, session):
        #self.batch_size = tf.placeholder(tf.int32, shape=[])
        #self.batch_sample = meta_data.build_input
        #self.n_data = meta_data.n_data
        self.batch_sample = meta_data.build_input
        self.sess = session

    def sample(self, quantity):
        return self.batch_sample(quantity)


    def get_exemple(self):
        return self.sample(1)

    def get_next_batch(self, quantity):
        pass
