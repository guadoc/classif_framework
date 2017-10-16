import tensorflow as tf
import tqdm as tq
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.python.client import timeline


class Train:
    def __init__(self, opts, model, monitor):
        self.optimizer = self.get_optimizer(opts)
        self.optim = self.optimizer.minimize(monitor.loss)
        self.regularizers = self.get_regularizers(model)
        self.save_model = opts.save_model


    def train(self, monitor, model):
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #self.valid(monitor, model)
        #monitor.end_epoch()
        for epoch in range(monitor.n_epoch - monitor.last_epoch):
            monitor.init_epoch()
            optim_param = model.optim_param(monitor)
            for bat in tq.tqdm(range(monitor.n_train_bat_epoch)):                
                # debug_time = time.time()
                run_metadata = tf.RunMetadata()
                _, _, metrics, summary = model.sess.run([self.optim,
                                        self.regularizers,
                                        monitor.train_metrics, 
                                        monitor.tr_batch_summary],
                                        feed_dict={
                                            monitor.batch_size: monitor.tr_batch_size,
                                            monitor.training_mode: True,
                                            self.lr: optim_param["lr"],
                                            self.momentum: optim_param["momentum"]
                                        })#, options=run_options, run_metadata=run_metadata)
                #print('Train time --------- [ '+str(time.time() - debug_time))
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('/net/phoenix/blot/timeline.json', 'w') as f:
                #     f.write(ctf)
                monitor.bat_train_update(metrics, summary)
                # print('Update time ##### '+str(time.time() - debug_time))

            self.valid(monitor, model)
            monitor.end_epoch()
            if self.save_model:
                model.model_save(monitor.epoch)



    def valid(self, monitor, model):
        for bat in tq.tqdm(range(monitor.n_val_batch)):
            metrics = model.sess.run(monitor.val_metrics, feed_dict={
                                                                    monitor.batch_size:monitor.val_batch_size,
                                                                    monitor.training_mode: False})
            monitor.bat_val_update(metrics)

    def get_optimizer(self, opts):
        if opts.optim =='adam':
            print("## adam optim")
            self.lr = tf.placeholder(tf.float32, shape=[])
            param = {"learning_rate":self.lr}
            return tf.train.AdamOptimizer(**param)
        elif opts.optim =='sgd':
            print("## sgd optim")
            self.lr = tf.placeholder(tf.float32, shape=[])
            self.momentum = tf.placeholder(tf.float32, shape=[])
            param = {"learning_rate":self.lr, "momentum":self.momentum}
            return tf.train.MomentumOptimizer(**param)


    def get_regularizers(self, model):
        regs = []
        regul_list = model.layer_regularizer()
        for i in range(len(regul_list)):
            reg = regul_list[i][0]
            var = regul_list[i][1]
            regs.append(self.optimizer.minimize(reg , var_list=var))
        return regs
