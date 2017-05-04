import numpy as np
import tensorflow as tf
import math
from scipy import stats as st
import numpy
import matplotlib.pyplot as plt
import os
import time

class Monitor:
    def __init__(self, opts, model):
        #general
        #self.log = opts.log
        #self.model = opts.model
        self.sess = model.sess
        #training parameters
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.tr_batch_size = opts.batch_size
        self.epoch = self.last_epoch
        self.n_train_bat_epoch = math.floor(opts.n_data_train / self.tr_batch_size)
        self.batch = self.n_train_bat_epoch * self.last_epoch
        #self.train_infos = np.array(np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.train_infos = [np.array([], dtype=np.int64) for x in range(4)]

        # validation parameters
        self.val_batch_size = self.tr_batch_size
        self.n_val_batch = math.floor(opts.n_data_val / self.val_batch_size)


        if opts.last_epoch > 0:
            pass #load log data

        self.train_metrics = {
                        'infos':model.infos,
                        'loss':model.loss,
                        'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(model.outputs,1), tf.int32), tf.cast(model.labels, tf.int32)), tf.float32)),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   model.outputs, model.labels, k=5 )  ))
                        }
        self.val_metrics = {
                        'loss':model.loss,
                        'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(model.outputs,1), tf.int32), tf.cast(model.labels, tf.int32)), tf.float32)),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   model.outputs, model.labels, k=5 )  ))
                        }
        self.epoch_train_stat = {
                        'loss':0,
                        'accuracy_top1':0,
                        'accuracy_top5':0,
                        }
        self.epoch_val_stat = {
                        'loss':0,
                        'accuracy_top1':0,
                        'accuracy_top5':0,
                        }

        tr_loss_summary = tf.summary.scalar('train loss', model.loss, collections=['per_batch'])
        self.tr_epoch_loss = tf.placeholder(tf.float32, shape=[])
        self.tr_epoch_accuracy_top1 = tf.placeholder(tf.float32, shape=[])
        self.tr_epoch_accuracy_top5 = tf.placeholder(tf.float32, shape=[])
        self.val_epoch_loss = tf.placeholder(tf.float32, shape=[])
        self.val_epoch_accuracy_top1 = tf.placeholder(tf.float32, shape=[])
        self.val_epoch_accuracy_top5 = tf.placeholder(tf.float32, shape=[])
        tr_epoch_loss_summary = tf.summary.scalar('epoch training loss', self.tr_epoch_loss, collections=['per_epoch'])
        tr_epoch_accuracy_top1_summary = tf.summary.scalar('epoch training accuracy_top1', self.tr_epoch_accuracy_top1, collections=['per_epoch'])
        tr_epoch_accuracy_top5_summary = tf.summary.scalar('epoch training accuracy_top5', self.tr_epoch_accuracy_top5, collections=['per_epoch'])
        val_epoch_loss_summary = tf.summary.scalar('epoch validation loss', self.val_epoch_loss, collections=['per_epoch'])
        val_epoch_accuracy_top1_summary = tf.summary.scalar('epoch validation accuracy_top1', self.val_epoch_accuracy_top1, collections=['per_epoch'])
        val_epoch_accuracy_top5_summary = tf.summary.scalar('epoch validation accuracy_top5', self.val_epoch_accuracy_top5, collections=['per_epoch'])

        self.tr_batch_summary = tf.summary.merge_all(key='per_batch')
        self.tr_epoch_summary = tf.summary.merge_all(key='per_epoch')
        self.log_writer = tf.summary.FileWriter(opts.log)
        self.epoch_time = time.time()

    def use_infos(self, infos):
        for i in range(len(infos)):
            self.train_infos[i] =  np.concatenate((self.train_infos[i], infos[i]))



    def bat_train_update(self, metrics, summary):
        self.use_infos(metrics['infos'])
        self.batch += 1
        self.epoch_train_stat['loss'] += metrics['loss']
        self.epoch_train_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_train_stat['accuracy_top5'] += metrics['accuracy_top5']
        self.log_writer.add_summary(summary, self.batch)

    def bat_val_update(self, metrics):
        self.epoch_val_stat['loss'] += metrics['loss']
        self.epoch_val_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_val_stat['accuracy_top5'] += metrics['accuracy_top5']

    def init_epoch(self):
        self.train_infos = [np.array([], dtype=np.int64) for x in range(4)]

        self.epoch = self.epoch + 1
        self.epoch_val_stat['loss'] = 0
        self.epoch_val_stat['accuracy_top1'] = 0
        self.epoch_val_stat['accuracy_top5'] = 0
        self.epoch_train_stat['loss'] = 0
        self.epoch_train_stat['accuracy_top1'] = 0
        self.epoch_train_stat['accuracy_top5'] = 0
        self.epoch_time = time.time()

    def end_epoch(self):
        #computing metrics
        train_accuracy_top1 = 100*self.epoch_train_stat['accuracy_top1']/(self.n_train_bat_epoch*self.tr_batch_size)
        train_accuracy_top5 = 100*self.epoch_train_stat['accuracy_top5']/(self.n_train_bat_epoch*self.tr_batch_size)
        train_loss = self.epoch_train_stat['loss']/self.n_train_bat_epoch
        val_accuracy_top1 = 100*self.epoch_val_stat['accuracy_top1']/(self.n_val_batch*self.val_batch_size) or 0
        val_accuracy_top5 = 100*self.epoch_val_stat['accuracy_top5']/(self.n_val_batch*self.val_batch_size) or 0
        val_loss = self.epoch_val_stat['loss']/self.n_val_batch
        epoch_summary = self.sess.run(self.tr_epoch_summary, feed_dict={self.tr_epoch_loss: train_loss,
                                                                        self.tr_epoch_accuracy_top1: train_accuracy_top1,
                                                                        self.tr_epoch_accuracy_top5: train_accuracy_top5,
                                                                        self.val_epoch_loss: val_loss,
                                                                        self.val_epoch_accuracy_top1: val_accuracy_top1,
                                                                        self.val_epoch_accuracy_top5: val_accuracy_top5})
        self.log_writer.add_summary(epoch_summary, self.epoch)

        #printing report
        m, s = divmod(time.time() - self.epoch_time, 60)
        h, m = divmod(m, 60)
        print('Epoch %d[%d:%d:%d], Test[loss: %.3f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%], Train[loss: %.3f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (self.epoch,
                h, m, s,
                val_loss,
                val_accuracy_top1,
                val_accuracy_top5,
                train_loss,
                train_accuracy_top1,
                train_accuracy_top5))


        hi1 = numpy.histogram( self.train_infos[0], bins=1000, density=1, range=(-3, 3))[0]
        hi2 = numpy.histogram( self.train_infos[1], bins=1000, density=1, range=(-3, 3))[0]
        hi3 = numpy.histogram( self.train_infos[2], bins=1000, density=1, range=(-3, 3))[0]
        hi4 = numpy.histogram( self.train_infos[3], bins=1000, density=1, range=(-3, 3))[0]
        h1 = st.entropy(hi1 / np.sum(hi1) )
        h2 = st.entropy(hi2 / np.sum(hi2) )
        h3 = st.entropy(hi3 / np.sum(hi3) )
        h4 = st.entropy(hi4 / np.sum(hi4) )
        print('h1: %.4f, h2: %.4f, h3: %.4f, h4: %.4f'% (h1, h2, h3, h4))
