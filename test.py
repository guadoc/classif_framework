

import tensorflow as tf
import importlib
from data import Data
from monitor_test import Monitor
from train import Train
from model import Model
import tqdm as tq
import numpy as np


#construction of the model
# models_config <-- [index, epoch]
#models_config = [[603, 500], [0,0]]
models_config = [[1, 1], [0,0]]
from imagenet.config_test import init_test_config
opts = init_test_config(models_config[0])

metadata = importlib.import_module(opts.dataset+'.metadata')
data = metadata.Metadata('ATEST', opts)

test_model = Model(opts, data)


#data loaders
test_set = Data(data, test_model.sess) # somewhere here we're running stuff
monitor = Monitor(opts, test_model, test_set)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=test_model.sess, coord=coord)

#initilization of all variables
test_model.initialize_variables(opts)


for bat in tq.tqdm(range(monitor.n_test_batch)):
    metrics = test_model.sess.run(monitor.test_metrics,
                                             feed_dict={
                                                monitor.batch_size: monitor.test_batch_size,
                                                monitor.training_mode: False
                                            })
    monitor.bat_test_update(metrics)
monitor.end_test()


coord.request_stop()
coord.join(threads)
test_model.sess.close()
