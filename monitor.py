import numpy as np
import tensorflow as tf
import math
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

        # validation parameters
        self.val_batch_size = self.tr_batch_size
        self.n_val_batch = math.floor(opts.n_data_val / self.val_batch_size)


        if opts.last_epoch > 0:
            pass #load log data

        self.train_metrics = {
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


    def bat_train_update(self, metrics, summary):
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
        print('Epoch %d[%d:%d:%d], Test[loss: %.2f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%], Train[loss: %.2f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (self.epoch,
                h, m, s,
                val_loss,
                val_accuracy_top1,
                val_accuracy_top5,
                train_loss,
                train_accuracy_top1,
                train_accuracy_top5))
