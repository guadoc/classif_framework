
import tensorflow as tf
import importlib
from data import Data
from monitor import Monitor
from config import init_config
from model import Model
import tqdm as tq
import time
import numpy as np
#import Image
from PIL import Image
import matplotlib.pyplot as plt

opts = init_config()
metadata = importlib.import_module(opts.dataset+'.metadata')




#with tf.device('/gpu:0')
#initialization of datasets
#training en validation datasets
train_meta_data = metadata.Metadata('train', opts)
#val_meta_data = metadata.Metadata('val', opts)

#initialization of monitoring and controler of training


#construction of the model
train_model = Model(opts, train_meta_data)
monitor = Monitor(opts, train_model.sess)
train_set = Data(train_meta_data, train_model.sess, monitor)
#val_set = Data(val_meta_data, train_model.sess, monitor)
mean = 0




for bat in tq.tqdm(range(monitor.n_train_bat_epoch)):
    batch_sample = train_set.sample(monitor.tr_batch_size)
    #img = Image.fromarray(batch_sample[0][0], 'RGB')
    #img.show()
