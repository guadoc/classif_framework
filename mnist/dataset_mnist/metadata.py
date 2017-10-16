
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import stats
import math

class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        self.datapath = opts.data
        mnist = input_data.read_data_sets('mnist/dataset_mnist/MNIST_data', one_hot=False)
        if data_type == 'train':
            self.image_number = 60000
            self.data = mnist.train
        else:
            self.image_number = 10000
            self.data = mnist.test
        self.image_size = 28
        self.model_size = self.image_size * self.image_size
        self.input_shape = [self.model_size]
        self.input_bat_shape = [None, self.model_size]
        self.label_bat_shape = [None]

        self.sess = tf.Session()


    def build_input(self, batch_size):
        batch_size= 100
        batch = self.data.next_batch(batch_size)#bug to check later
        inputs = batch[0]#(batch[0]-0.5)*2.
        labels = batch[1]
        return {"inputs":inputs , "labels":labels}
