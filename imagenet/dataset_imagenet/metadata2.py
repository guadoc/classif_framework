
import tensorflow as tf
import os
import time
import re
import numpy as np
from imagenet.dataset_imagenet.synset import *
from tensorflow.python.ops import control_flow_ops
#IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]


class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        self.datapath = os.path.join(opts.data, data_type)
        self.n_classes = 1000
        self.image_size = 448
        self.model_size = 448
        self.model_depth = 3
        self.input_bat_shape = [None, self.model_size, self.model_size, self.model_depth]
        self.label_bat_shape = [None]
        self.input_shape = [self.model_size, self.model_size, self.model_depth]
        self.image_shape = [self.image_size, self.image_size]



    def file_list(self, data_dir):
        filenames = []
        counter = 0
        for dir_ in os.listdir(data_dir):
            for file_ in os.listdir(os.path.join(data_dir, dir_)):
                if os.path.isfile(os.path.join(data_dir, dir_, file_)):
                    line = file_.rstrip()
                    fn = os.path.join(data_dir, dir_, line)
                    filenames.append(fn)
                    counter += 1
        return filenames, counter


    def load_data(self, data_dir):
        data = []
        i = 0
        print("-- listing files in", data_dir)
        start_time = time.time()
        files, self.image_number = self.file_list(data_dir)
        for img_fn in files:
            ext = os.path.splitext(img_fn)[1]
            if ext != '.JPEG': self.image_number -= 1

            label_name = re.search(r'(n\d+)', img_fn).group(1)
            fn = os.path.join(data_dir, img_fn)

            label_index = synset_map[label_name]["index"]

            data.append({
                "filename": fn,
                "label_name": label_name,
                "label_index": label_index,
                "desc": synset[label_index],
            })
        duration = time.time() - start_time
        print ("## took %f sec, load %d filenames for %s dataset" % (duration, self.image_number, self.data_type))
        return data


    def build_input(self, batch_size):
        data = self.load_data(self.datapath)
        filenames = [ d['filename'] for d in data ]
        label_indexes = [ d['label_index'] for d in data ]
        # Up to here, nothing is symbolic
        filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=(self.data_type == 'train'))
        image_file = tf.read_file(filename)
        image_data = tf.image.decode_jpeg(image_file, channels=3)
        #image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        image = self.image_preprocessing(image_data)
        images, labels = tf.train.batch([image, label_index], batch_size=batch_size, num_threads=120, enqueue_many=False, capacity=500, allow_smaller_final_batch=True)        
        return {"inputs":images, "labels":labels}


    def image_preprocessing(self, image):
        height = self.model_size
        width = self.model_size
        rescaled_image = tf.expand_dims(image, 0)
        rescaled_image = tf.image.resize_bilinear(rescaled_image, [height, width], align_corners=False)
        rescaled_image = tf.squeeze(rescaled_image, [0])
        if self.data_type =='train':
            rescaled_image = tf.image.random_flip_left_right(rescaled_image)
        return rescaled_image




