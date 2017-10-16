import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import time




class Monitor:

    def __init__(self, opts, model, test_set):
        #general

        self.test_set = test_set

        #training parameters
        #self.val_infos   = [np.array([], dtype=np.int64),   np.array([], dtype=np.int64)]

        # validation parameters
        self.test_batch_size = opts.batch_size
        #self.n_test_batch = math.floor(test_set.n_data / self.test_batch_size)
        self.n_test_batch = math.floor(opts.n_data_test / self.test_batch_size)

        self.training_mode = tf.placeholder(tf.bool, shape=[])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.test_sample = self.test_set.sample(self.batch_size)

        self.inputs = self.test_sample["inputs"]
        self.labels = self.test_sample["labels"]
        self.outputs, infos = model.inference(self.inputs, self.training_mode)
        #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels))

        self.n_test_crop = 10
        outs = []#self.outputs
        for i in range(self.test_batch_size):
            outs.append(tf.reduce_mean(self.outputs[i*self.n_test_crop:(i+1)*self.n_test_crop], 0))

        self.test_metrics = {
                        'accuracy_top1':tf.reduce_sum(tf.to_float(tf.equal(tf.cast(tf.argmax(outs, 1), tf.int32), tf.cast(self.labels, tf.int32)))),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(outs, self.labels, k=5 )  ) )
                        }
        self.epoch_test_stat = {
                        'accuracy_top1':0,
                        'accuracy_top5':0
                        }



    def bat_test_update(self, metrics):
        self.epoch_test_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_test_stat['accuracy_top5'] += metrics['accuracy_top5']


    def end_test(self):
        test_accuracy_top1 = 100*self.epoch_test_stat['accuracy_top1']/(self.n_test_batch*self.test_batch_size) or 0
        test_accuracy_top5 = 100*self.epoch_test_stat['accuracy_top5']/(self.n_test_batch*self.test_batch_size) or 0
        print('Test summary [accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (test_accuracy_top1,
                test_accuracy_top5))
