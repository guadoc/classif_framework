import tensorflow as tf
import tqdm as tq
import numpy as np
from valid import Valid
import matplotlib.pyplot as plt
import time



class Train:
    def __init__(self, opts, model, train_set, val_set):
        self.optimizer = self.optimizer_(opts)
        self.optim = self.optimizer.minimize(model.loss)
        self.regularizers = self.regularizers_(model)
        #self.optim = self.optimizer.minimize(model.regularizer + model.loss)
        self.valider = Valid(opts, model, val_set)
        self.data = train_set
        self.save_model = opts.save_model

    def train(self, monitor, model):
        for epoch in range(monitor.n_epoch - monitor.last_epoch):
            monitor.init_epoch()
            optim_param = model.optim_param(monitor)
            for bat in tq.tqdm(range(monitor.n_train_bat_epoch)):
                batch = self.data.sample(monitor.tr_batch_size)
                _, _, metrics, summary = model.sess.run([self.regularizers,
                                                    self.optim,
                                                    monitor.train_metrics,
                                                    monitor.tr_batch_summary],
                                                    feed_dict={
                                                        model.inputs: batch["inputs"],
                                                        model.labels: batch["labels"],
                                                        model.training_mode: True,
                                                        self.lr: optim_param["lr"],
                                                        self.momentum: optim_param["momentum"]
                                                    })
                monitor.bat_train_update(metrics, summary)
            self.valider.valid(monitor, model)
            monitor.end_epoch()
            if self.save_model:
                model.model_save(monitor.epoch)


    def optimizer_(self, opts):
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

    def regularizers_(self, model):
        regs = []
        regul_list = model.layer_regularizer

        for i in range(len(regul_list)):
            reg = regul_list[i][0][0]
            var = regul_list[i][1]
            regs.append(self.optimizer.minimize(reg , var_list=var))
        return regs
