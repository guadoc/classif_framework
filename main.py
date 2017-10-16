
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import importlib
from data import Data
from monitor import Monitor
from train import Train
from model import Model

from cifar.config import init_config
opts = init_config()

#training and validation datasets
metadata = importlib.import_module(opts.dataset+'.metadata')
train_meta_data = metadata.Metadata('train', opts)
val_meta_data   = metadata.Metadata('val', opts)

#construction of the model
train_model = Model(opts, train_meta_data)
#data loaders
train_set = Data(train_meta_data, train_model.sess)
val_set = Data(val_meta_data, train_model.sess)

#trainer modules
monitor = Monitor(opts, train_model, train_set, val_set)
trainer = Train(opts, train_model, monitor)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=train_model.sess, coord=coord)

#initilization of all variables
train_model.initialize_variables(opts)


#training
trainer.train(monitor, train_model)
