
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
        self.image_size = 256
        self.model_size = 224
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
            im_name = os.path.split(img_fn)[1]
            label_index = synset_map[label_name]["index"]            

            data.append({
                "filename": img_fn,
                "label_name": label_name,
                "im_name": im_name,
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
        im_names = [ d['im_name'] for d in data ]
        lab_names = [ d['label_name'] for d in data ]
        # Up to here, nothing is symbolic
        filename, label_index, image_name, label_name = tf.train.slice_input_producer([filenames, label_indexes, im_names, lab_names], shuffle=True)#(self.data_type == 'train'))
        image_file = tf.read_file(filename)
        image_deco = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
        image_data = tf.image.convert_image_dtype(image_deco, dtype=tf.float32)
        image = self.image_preprocessing(image_deco)
        images, labels, image_names, label_names = tf.train.batch([image, label_index, image_name, label_name], batch_size=batch_size, num_threads=64, enqueue_many=False, capacity=2000, allow_smaller_final_batch=True)
        if self.data_type == 'test':
            images = self.inputs_augment(images, batch_size)
        return {"inputs":images, "labels":labels, "label_names": label_names, "image_names": image_names}


    def inputs_augment(self, inputs, batch_size):
        #inputs_batch = tf.tile(inputs, [0,1,1,1])
        inputs_batch = tf.zeros([0, self.model_size, self.model_size, self.model_depth], dtype=tf.float32)
        i = tf.constant(0)

        def cond(i, inputs_batch):
            return tf.less(i, batch_size)

        def body(i, inputs_batch):
            im = inputs[i]
            im1 = tf.image.crop_to_bounding_box(im, 0, 0, self.model_size, self.model_size)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(im1, 0)], 0)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(tf.image.flip_left_right(im1), 0)], 0)
            im1 = tf.image.crop_to_bounding_box(im, 0, self.image_size-self.model_size, self.model_size, self.model_size)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(im1, 0)], 0)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(tf.image.flip_left_right(im1), 0)], 0)
            im1 = tf.image.crop_to_bounding_box(im, self.image_size-self.model_size, 0, self.model_size, self.model_size)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(im1, 0)], 0)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(tf.image.flip_left_right(im1), 0)], 0)
            im1 = tf.image.crop_to_bounding_box(im, self.image_size-self.model_size, self.image_size-self.model_size, self.model_size, self.model_size)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(im1, 0)], 0)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(tf.image.flip_left_right(im1), 0)], 0)
            center_ind = tf.cast(tf.floor(tf.to_float(self.image_size-self.model_size)/2), tf.int32)
            im1 = tf.image.crop_to_bounding_box(im, center_ind, center_ind, self.model_size, self.model_size)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(im1, 0)], 0)
            inputs_batch = tf.concat([inputs_batch, tf.expand_dims(tf.image.flip_left_right(im1), 0)], 0)
            return tf.add(i, 1), inputs_batch
        # do the loop:
        i, inputs_batch = tf.while_loop(cond, body, [i, inputs_batch], shape_invariants=[i.get_shape(), tf.TensorShape([None, self.model_size, self.model_size, self.model_depth])])#inputs.get_shape()])
        return inputs_batch



    def image_preprocessing(self, image):
        height = self.model_size
        width = self.model_size
        if self.data_type =='train':
            image = distort_image(image, height, width)
        elif self.data_type =='val':
            image = eval_image(image, height, width)
        else:
            image =  eval_image(image, height, width)
            #image = eval_image(image, self.image_size, self.image_size)
            #test images
        return image

def distort_color(image):
    color_ordering=0
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        pass
    return image


def distort_image(image, height, width):
    shape = tf.shape(image)
    min_scale = tf.random_uniform([], minval=256, maxval=480, dtype=tf.float32 )
    max_ratio = tf.cast(shape[1], tf.float32) / tf.cast(shape[0], tf.float32)
    scaled_height, scaled_width = control_flow_ops.cond(shape[0] < shape[1], lambda: (min_scale, min_scale*max_ratio ), lambda: (min_scale/max_ratio , min_scale))
    distorted_image = image
    distorted_image = tf.expand_dims(distorted_image, 0)
    distorted_image = tf.image.resize_bilinear(distorted_image, [tf.cast(scaled_height, tf.int32), tf.cast(scaled_width, tf.int32)], align_corners=False)
    distorted_image = tf.squeeze(distorted_image, [0])
    distorted_image = tf.random_crop(distorted_image, [height, width, 3])    
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image)
    return distorted_image


def eval_image_(image, height, width, scope=None):
    shape = tf.shape(image)
    short_edge = tf.reduce_min(shape[:2])
    #short_edge = tf.Print(short_edge, [short_edge, tf.shape(image)[0], tf.shape(image)[1]])
    crop_img = tf.image.resize_image_with_crop_or_pad(image, short_edge, short_edge)
    resized_img = tf.image.resize_images(crop_img, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    return resized_img


def eval_image(image, height, width, scope=None):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225] 
    im = image
    shape = tf.shape(im)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    height_smaller_than_width = tf.less_equal(height, width)
    new_shorter_edge = tf.constant(256.0)
    new_height, new_width = control_flow_ops.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, new_shorter_edge * width / height ),
        lambda: (new_shorter_edge * height / width , new_shorter_edge))
    im = tf.image.resize_images(im, [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)], method=tf.image.ResizeMethod.BILINEAR)
    im = tf.image.resize_image_with_crop_or_pad(im, 224, 224)
    im = im - mean
    im = im / std
    return im
