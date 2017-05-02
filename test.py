

import tensorflow as tf
import importlib
from data import Data
from monitor_test import Monitor
from train import Train
from model import Model
import tqdm as tq
import numpy as np

from imagenet.config import init_config
opts = init_config()

#training and validation datasets
metadata = importlib.import_module(opts.dataset+'.metadata')
val_meta_data = metadata.Metadata('test', opts)

#construction of the model
test_model = Model(opts, val_meta_data)
#data loaders
test_set = Data(val_meta_data, test_model.sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=test_model.sess, coord=coord)

monitor = Monitor(opts, test_model)

#initilization of all variables
test_model.initialize_variables(opts)


for bat in tq.tqdm(range(monitor.n_test_batch)):
    batch = test_set.sample(monitor.test_batch_size)
    metrics = test_model.sess.run(monitor.test_metrics, feed_dict={test_model.inputs: np.asarray(batch["inputs"]), test_model.labels: batch["labels"], test_model.training_mode: False})
    monitor.bat_test_update(metrics)

monitor.end_test()
