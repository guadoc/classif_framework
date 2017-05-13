
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math

class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        self.datapath = opts.data
        mnist = input_data.read_data_sets('mnist/dataset_mnist/MNIST_data', one_hot=True)
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


    def build_input(self, batch_size):
        batch_size=600
        batch = self.data.next_batch(batch_size)#bug to check later        
        return {"inputs":batch[0], "labels":np.argmax(batch[1], axis=1)}
