import numpy as np
import tensorflow as tf
import math
from scipy import stats as st
import matplotlib.pyplot as plt
import os
import time
import numpy.ma as ma
from tensorflow.python.ops import control_flow_ops

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


import importlib

class Monitor:
    def __init__(self, opts, model, train_set, val_set):
        #general
        self.train_set = train_set
        self.val_set = val_set
        self.sess = model.sess
        self.save = opts.save_logs
        #training parameters
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.tr_batch_size = opts.batch_size
        self.epoch = self.last_epoch
        self.n_train_bat_epoch = math.floor(opts.n_data_train / self.tr_batch_size)
        self.batch = self.n_train_bat_epoch * self.last_epoch
        self.train_infos = [np.array([], dtype=np.int64),   np.array([], dtype=np.int64)]
        self.val_infos   = [np.array([], dtype=np.int64),   np.array([], dtype=np.int64)]
        self.checkpoint = opts.checkpoint # or 1
        # validation parameters
        self.val_batch_size = self.tr_batch_size
        self.n_val_batch = math.floor(opts.n_data_val / self.val_batch_size)


        if opts.last_epoch > 0:
            pass #load log data

        self.training_mode = tf.placeholder(tf.bool, shape=[])        
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        zero_batch_size = tf.constant(0)


        val_batch_size, train_batch_size = control_flow_ops.cond(self.training_mode, lambda: (zero_batch_size, self.batch_size) , lambda: (self.batch_size, zero_batch_size ))
        train_sample = self.train_set.sample(train_batch_size)
        val_sample = self.val_set.sample(val_batch_size)
        sample = control_flow_ops.cond(self.training_mode, lambda: train_sample, lambda: val_sample)
        self.inputs = sample["inputs"]
        self.labels = sample["labels"]
        self.labels = tf.Print(self.labels, [self.labels])
        self.outputs, _ = model.inference(self.inputs, self.training_mode)                
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels))


        self.train_metrics = {
                        #'infos':[self.infos, self.train_sample["labels"]],
                        'loss':self.loss,
                        'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(self.outputs,1), tf.int32), tf.cast(self.labels, tf.int32)), tf.float32)),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   self.outputs, self.labels, k=5 )  ))
                        }
        self.val_metrics = {
                        #'infos':[model.infos, model.labels],
                        'loss':self.loss,
                        'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(self.outputs,1), tf.int32), tf.cast(self.labels, tf.int32)), tf.float32)),
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   self.outputs, self.labels, k=5 )  ))
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

        tr_loss_summary = tf.summary.scalar('train loss', self.loss, collections=['per_batch'])
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

        self.tr_batch_summary = tf.summary.merge_all(key='per_batch')#tf.summary.merge_all()#
        self.tr_epoch_summary = tf.summary.merge_all(key='per_epoch')
        self.log_writer = tf.summary.FileWriter(opts.log)#, graph=tf.get_default_graph())
        self.epoch_time = time.time()



    def use_train_infos(self, infos):
        if self.train_infos[0].shape[0] == 0:
            self.train_infos[0] = infos[0]
            self.train_infos[1] = infos[1]
        else:
            self.train_infos[0] = np.concatenate((self.train_infos[0], infos[0]), axis=0)
            self.train_infos[1] = np.concatenate((self.train_infos[1], infos[1]), axis=0)


    def use_val_infos(self, infos):
        if self.val_infos[0].shape[0] == 0:
            self.val_infos[0] = infos[0]
            self.val_infos[1] = infos[1]
        else:
            self.val_infos[0] = np.concatenate((self.val_infos[0], infos[0]), axis=0)
            self.val_infos[1] = np.concatenate((self.val_infos[1], infos[1]), axis=0)


    def bat_train_update(self, metrics, summary):
        #self.use_train_infos(metrics['infos'])
        self.batch += 1
        self.epoch_train_stat['loss'] += metrics['loss']
        self.epoch_train_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_train_stat['accuracy_top5'] += metrics['accuracy_top5']
        if self.save and self.batch%self.checkpoint==0:
            self.log_writer.add_summary(summary, self.batch)

    def bat_val_update(self, metrics):
        #self.use_val_infos(metrics['infos'])
        self.epoch_val_stat['loss'] += metrics['loss']
        self.epoch_val_stat['accuracy_top1'] += metrics['accuracy_top1']
        self.epoch_val_stat['accuracy_top5'] += metrics['accuracy_top5']

    def init_epoch(self):
        self.train_infos = [np.array([], dtype=np.int64),   np.array([], dtype=np.int64)]
        self.val_infos   = [np.array([], dtype=np.int64),   np.array([], dtype=np.int64)]

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
        if self.save:
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
        print('Epoch %d[%d:%d:%d], Test[loss: %.3f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%], Train[loss: %.4f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (self.epoch,
                h, m, s,
                val_loss,
                val_accuracy_top1,
                val_accuracy_top5,
                train_loss,
                train_accuracy_top1,
                train_accuracy_top5))

        # val_ent = entropy(self.val_infos[0])
        # tr_ent = entropy(self.train_infos[0])
        # val_cond_ent = entropy_cond(self.val_infos[0], self.val_infos[1]) /val_ent
        # tr_cond_ent = entropy_cond(self.train_infos[0], self.train_infos[1]) / tr_ent
        # print("Entropy:              Test[%.5f],  Train[%.5f]"%(val_ent, tr_ent))
        # print("Conditionnal Entropy: Test[%.5f],  Train[%.5f]"%(val_cond_ent, tr_cond_ent))

def entropy(outputs):
    histos = np.apply_along_axis(lambda a: np.histogram(a, bins=100, density=1, range=(-1,1))[0], 0, outputs)
    entropys = np.apply_along_axis(lambda a: st.entropy(a / np.sum(a)), 0, histos)
    return np.sum(entropys)

def entropy_cond(outputs, labels):
    nlabel = 10
    ent = np.zeros(nlabel)
    for i in range(nlabel):
        mask = np.array(labels==i)
        out = outputs[mask]
        ent[i] = entropy(out)
    return np.mean(ent)
