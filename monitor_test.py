import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import time

class Monitor:
    def __init__(self, opts, model):
        #general
        self.sess = model.sess
s
        # testing parameters
        self.test_batch_size = 10#opts.batch_size
        self.n_test_batch = math.floor(opts.n_data_val / self.test_batch_size)
        self.n_test_data_augment = 10

        outs = []
        for i in range(self.test_batch_size):
            outs.append(tf.reduce_mean(model.outputs[i*self.n_test_data_augment:(i+1)*self.n_test_data_augment], 0))

        self.test_metrics = {
                        'accuracy_top1':tf.reduce_sum(tf.to_float(tf.equal(tf.cast(tf.argmax(outs, 1), tf.int32), tf.cast(model.labels, tf.int32)))),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(outs, model.labels, k=5 )  ) )
                        }
        self.epoch_test_stat = {
                        'activation':[]
                        'accuracy_top1':0,
                        'accuracy_top5':0
                        }

        tr_loss_summary = tf.summary.scalar('train loss', model.loss, collections=['per_batch'])
        self.tr_epoch_loss = tf.placeholder(tf.float32, shape=[])
        tr_epoch_loss_summary = tf.summary.scalar('epoch training loss', self.tr_epoch_loss, collections=['per_epoch'])

        self.tr_batch_summary = tf.summary.merge_all(key='per_batch')
        self.tr_epoch_summary = tf.summary.merge_all(key='per_epoch')
        self.log_writer = tf.summary.FileWriter(opts.log)



    def bat_test_update(self, metrics):
        self.epoch_test_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_test_stat['accuracy_top5'] += metrics['accuracy_top5']


    def end_test(self):
        test_accuracy_top1 = 100*self.epoch_test_stat['accuracy_top1']/(self.n_test_batch*self.test_batch_size) or 0
        test_accuracy_top5 = 100*self.epoch_test_stat['accuracy_top5']/(self.n_test_batch*self.test_batch_size) or 0
        print('Test summary [accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (test_accuracy_top1,
                test_accuracy_top5))
