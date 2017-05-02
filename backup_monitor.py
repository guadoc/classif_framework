import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import time

class Monitor:
    def __init__(self, opts, sess):
        #general
        self.log = opts.log
        self.model = opts.model

        #training parameters
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.tr_batch_size = opts.batch_size
        self.n_bat_chkpt = opts.checkpoint
        self.epoch = self.last_epoch + 1
        self.n_train_bat_epoch = math.floor(opts.n_data_train / self.tr_batch_size)
        self.batch = self.n_train_bat_epoch * self.last_epoch


        # validation parameters
        self.val_batch_size = self.tr_batch_size
        self.n_val_batch = math.floor(opts.n_data_val / self.val_batch_size)

        if opts.last_epoch > 0:
            pass #load log data

        self.epoch_train_stat = {
                'save': {
                        'loss':np.load(self.log+'/train_loss.npy').tolist() if opts.last_epoch > 0 else [],
                        'accuracy_top1':np.load(self.log+'/train_accuracy_top1.npy').tolist() if opts.last_epoch > 0 else []
                        #'accuracy_top5':np.load(self.log+'/train_accuracy_top5.npy').tolist() if opts.last_epoch > 0 else []
                        },
                'cum': {
                        'loss':0,
                        'accuracy_top1':0,
                        'accuracy_top5':0
                        },
        }
        self.epoch_val_stat = {
                'save': {
                        'loss':np.load(self.log+'/val_loss.npy').tolist() if opts.last_epoch > 0 else [],
                        'accuracy_top1':np.load(self.log+'/val_accuracy_top1.npy').tolist() if opts.last_epoch > 0 else []
                        #'accuracy_top5':np.load(self.log+'/val_accuracy_top5.npy').tolist() if opts.last_epoch > 0 else []
                        },
                'cum': {
                        'loss':0,
                        'accuracy_top1':0,
                        'accuracy_top5':0
                        },
        }


        #plot
        fig_ckpt = plt.figure(1)
        self.ax_ckpt = fig_ckpt.add_subplot(111)
        fig_epoch = plt.figure(2)
        self.ax_epoch = fig_epoch.add_subplot(111)
        self.epoch_time = time.time()

    def top_k_error(predictions, labels, k):
        batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / batch_size

    def train_metrics(self, model, opts):
        train_metrics = {
            'loss':model.loss,
            'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(model.outputs,1), tf.int32), tf.cast(model.labels, tf.int32)), tf.float32))
            #'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   model.outputs, model.labels, k=5 )  ))
            }
        tr_loss_summary = tf.summary.scalar('train loss', model.loss)
        self.tr_loss_summary = tf.summary.merge([tr_loss_summary])
        self.log_writer_batch = tf.summary.FileWriter(opts.log, mo
        return train_metrics, merged_summary



    def val_metrics(self, model):
        val_metrics = {
            'loss':model.loss,
            'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(model.outputs,1), tf.int32), tf.cast(model.labels, tf.int32)), tf.float32))
            #'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   model.outputs, model.labels, k=5 )  ))
            }
        return val_metrics


    def bat_train_update(self, metrics, summary):
        self.batch += 1        
        self.epoch_train_stat['cum']['loss'] += metrics['loss']
        self.epoch_train_stat['cum']['accuracy_top1'] += metrics['accuracy_top1']
        #self.epoch_train_stat['cum']['accuracy_top5'] += metrics['accuracy_top5']
        self.log_writer_batch.add_summary(summary, self.batch)



    def bat_val_update(self, metrics):
        self.epoch_val_stat['cum']['loss'] += metrics['loss']
        self.epoch_val_stat['cum']['accuracy_top1'] += metrics['accuracy_top1']
        #self.epoch_val_stat['cum']['accuracy_top5'] += metrics['accuracy_top5']


    def epoch_plot(self):
        plt.figure(2)
        self.ax_epoch.clear()
        plt.xlabel('#epoch')
        plt.ylabel('percentage')
        plt.grid(True)
        plt.plot(  range(len(   self.epoch_train_stat['save']['accuracy_top1']  ))   , self.epoch_train_stat['save']['accuracy_top1'], label="train accuracy (top1)" )
        #plt.plot(  range(len(   self.epoch_train_stat['save']['accuracy_top5']  ))   , self.epoch_train_stat['save']['accuracy_top5'], label='train accuracy (top5)' )
        plt.plot(  range(len(   self.epoch_val_stat['save']['accuracy_top1']    ))   , self.epoch_val_stat['save']['accuracy_top1'], label='val accuracy (top1)' )
        #plt.plot(  range(len(   self.epoch_val_stat['save']['accuracy_top5']    ))   , self.epoch_val_stat['save']['accuracy_top5'], label='val accuracy (top5)' )
        plt.legend()
        plt.pause(0.0000001)


    def end_epoch(self, metrics, model):
        #computing metrics
        train_accuracy_top1 = 100*self.epoch_train_stat['cum']['accuracy_top1']/(self.n_train_bat_epoch*self.tr_batch_size)
        #train_accuracy_top5 = 100*self.epoch_train_stat['cum']['accuracy_top5']/(self.n_train_bat_epoch*self.tr_batch_size)
        train_loss = self.epoch_train_stat['cum']['loss']/self.n_train_bat_epoch
        val_accuracy_top1 = 100*self.epoch_val_stat['cum']['accuracy_top1']/(self.n_val_batch*self.val_batch_size) or 0
        #val_accuracy_top5 = 100*self.epoch_val_stat['cum']['accuracy_top5']/(self.n_val_batch*self.val_batch_size) or 0
        val_loss = self.epoch_val_stat['cum']['loss']/self.n_val_batch


        #updating Logs
        self.epoch_train_stat['save']['loss'].append(train_loss)
        self.epoch_train_stat['save']['accuracy_top1'].append(train_accuracy_top1)
        #self.epoch_train_stat['save']['accuracy_top5'].append(train_accuracy_top5)
        self.epoch_val_stat['save']['loss'].append(val_loss)
        self.epoch_val_stat['save']['accuracy_top1'].append(val_accuracy_top1)
        #self.epoch_val_stat['save']['accuracy_top5'].append(val_accuracy_top5)

        #saving Logs and model
        model.model_save(self.epoch)
        np.save(self.log + '/train_loss', self.epoch_train_stat['save']['loss'])
        np.save(self.log + '/train_accuracy_top1', self.epoch_train_stat['save']['accuracy_top1'])
        #np.save(self.log + '/train_accuracy_top5', self.epoch_train_stat['save']['accuracy_top5'])
        np.save(self.log + '/val_loss', self.epoch_val_stat['save']['loss'])
        np.save(self.log + '/val_accuracy_top1', self.epoch_val_stat['save']['accuracy_top1'])
        #np.save(self.log + '/val_accuracy_top5', self.epoch_val_stat['save']['accuracy_top5'])

        #printing report
        self.epoch_plot()
        m, s = divmod(time.time() - self.epoch_time, 60)
        h, m = divmod(m, 60)
        print('Epoch %d[%d:%d:%d], Test[loss: %.2f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%], Train[loss: %.2f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (self.epoch,
                h, m, s,
                val_loss,
                val_accuracy_top1,
                0,#val_accuracy_top5,
                train_loss,
                train_accuracy_top1,
                0))#train_accuracy_top5))


        #reset of variables
        self.epoch = self.epoch + 1
        self.epoch_val_stat['cum']['loss'] = 0
        self.epoch_val_stat['cum']['accuracy_top1'] = 0
        #self.epoch_val_stat['cum']['accuracy_top5'] = 0
        self.epoch_train_stat['cum']['loss'] = 0
        self.epoch_train_stat['cum']['accuracy_top1'] = 0
        #self.epoch_train_stat['cum']['accuracy_top5'] = 0
        self.epoch_time = time.time()

        print('patch') #for tqdm issue
