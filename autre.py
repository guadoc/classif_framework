
import tensorflow as tf
#import tensorflow.contrib as contrib
import numpy as np
import math
import tqdm as tq
import importlib
from tensorflow.python.client import timeline

from data import Data
from imagenet.config import init_config
opts = init_config()
#training and validation datasets
metadata = importlib.import_module(opts.dataset+'.metadata2')
train_meta_data = metadata.Metadata('val', opts)

sess = tf.Session()
#data loaders
train_set = Data(train_meta_data, sess)





def get_image(path): 
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]      
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")/255
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [448, 448], align_corners=False)
    image = tf.squeeze(image, [0])
    return [image]

# inputs = get_image(path = './cat.jpg')




training_mode = tf.placeholder(tf.bool, shape=[])
batch_size = tf.placeholder(tf.int32, shape=[])

# sample = get_image(path = './cat.jpg')[0]
# sample = [sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample,sample]
# labels = tf.convert_to_tensor([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])

sample = train_meta_data.build_input(batch_size)
labels = sample["labels"]
inputs = sample["inputs"]


train_model = importlib.import_module("imagenet.models_imagenet.weldone_torch_")
outputs, _ = train_model.inference(inputs, training_mode)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
lr = tf.placeholder(tf.float32, shape=[])
momentum = tf.placeholder(tf.float32, shape=[])
param = {"learning_rate":lr, "momentum":momentum}
optimizer = tf.train.MomentumOptimizer(**param)

optim = optimizer.minimize(loss)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.global_variables_initializer())



#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

for bat in tq.tqdm(range(100)):   
	#run_metadata = tf.RunMetadata()
	out = sess.run([outputs, optim], feed_dict={training_mode:True, batch_size:16, lr:0.001, momentum:0.9})# , options=run_options, run_metadata=run_metadata)
	# tl = timeline.Timeline(run_metadata.step_stats)
	# ctf = tl.generate_chrome_trace_format()
	# with open('/net/phoenix/blot/timeline.json', 'w') as f:
	#     f.write(ctf)


print(out[0].shape)
