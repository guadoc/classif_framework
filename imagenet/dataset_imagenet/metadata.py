
import tensorflow as tf
import os
import time
import re
import numpy as np
from imagenet.dataset_imagenet.synset import *
#IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        if data_type == 'train':
            self.datapath = os.path.join(opts.data, 'train')
        else:
            self.datapath = os.path.join(opts.data, 'val')
        self.n_classes = 1000
        self.image_size = 256
        self.model_size = 224
        self.model_depth = 3
        self.input_bat_shape = [None, self.model_size, self.model_size, self.model_depth]
        self.label_bat_shape = [None]
        self.input_shape = [self.model_size, self.model_size, self.model_depth]
        self.image_shape = [self.image_size, self.image_size]

    def file_list_from_txt(self, data_dir):
        dir_txt = data_dir + ".txt"
        filenames = []
        with open(dir_txt, 'r') as f:
            for line in f:
                if line[0] == '.': continue
                line = line.rstrip()
                fn = os.path.join(data_dir, line)
                filenames.append(fn)
        return filenames


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
        #filename = tf.Print(filename_, [filename_])
        image_file = tf.read_file(filename)
        image_deco = tf.image.decode_jpeg(image_file, channels=3)

        image_data = tf.image.resize_images(image_deco, self.image_shape)
        #tf.summary.image('image_data', image_data)

        if self.data_type == 'train':
            image = self.image_preprocessing(image_data)
            images, labels = tf.train.batch([image, label_index], batch_size=batch_size, num_threads=16, enqueue_many=False, capacity=2000, allow_smaller_final_batch=True)
        elif self.data_type == 'val':
            image = self.image_preprocessing(image_data)
            images, labels = tf.train.batch([image, label_index], batch_size=batch_size, num_threads=16, enqueue_many=False, capacity=2000, allow_smaller_final_batch=True)
        else:
            test_batch = self.test_batch(image_data)
            images, labels = tf.train.batch([test_batch, label_index], batch_size=batch_size, num_threads=16, enqueue_many=False, capacity=2000, allow_smaller_final_batch=True)
            images = self.concat_test_batch(images)
            #labels = [label_index]
            #images = self.test_batch(image_data)
        #tf.summary.image('mesimages', images)
        return {"inputs":images, "labels":labels}


    def image_preprocessing(self, image):
        image = tf.cast(image, tf.float32)
        #image = decode_jpeg(image_buffer)
        height = self.model_size
        width = self.model_size
        if self.data_type =='train':
            image = distort_image(image, height, width)
            #image = eval_image(image, height, width)
        else:
            image = eval_image(image, height, width)
        image = tf.subtract(image, 115)
        # Finally, rescale to [-1,1] instead of [0, 1)
        #image = self.unitize(image)
        return image

    def unitize(self, image):
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.subtract(image, 0.5)
        return tf.multiply(image, 2.0)

    def concat_test_batch(self, images):
        batch = []
        for bat in range(10):
            for i in range(10):
                batch.append(images[bat][i])
        return batch

    def test_batch(self, image):
        batch = []
        image_shape = tf.shape(image)
        image = tf.subtract(image, 115)
        im1 = tf.image.crop_to_bounding_box(image, 0, 0, self.model_size, self.model_size)
        batch.append( im1 )
        batch.append(tf.image.flip_left_right(im1))
        im2 = tf.image.crop_to_bounding_box(image, 0, image_shape[1]-self.model_size, self.model_size, self.model_size)
        batch.append( im2 )
        batch.append(tf.image.flip_left_right(im2))
        im3 = tf.image.crop_to_bounding_box(image, image_shape[0]-self.model_size, 0, self.model_size, self.model_size)
        batch.append( im3 )
        batch.append(tf.image.flip_left_right(im3))
        im4 = tf.image.crop_to_bounding_box(image, image_shape[0]-self.model_size, image_shape[1]-self.model_size, self.model_size, self.model_size)
        batch.append( im4 )
        batch.append(tf.image.flip_left_right(im4))
        center_ind = tf.cast(tf.floor(tf.to_float(image_shape[0]-self.model_size)/2), tf.int32)
        im5 = tf.image.crop_to_bounding_box(image, center_ind, center_ind, self.model_size, self.model_size)
        batch.append( im5 )
        batch.append(tf.image.flip_left_right(im5))
        return batch

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
    scale = np.random.uniform(1, 2)
    scaled_height = int(np.floor(scale*height))
    distorted_image = image
    distorted_image = tf.expand_dims(distorted_image, 0)
    distorted_image = tf.image.resize_bilinear(distorted_image, [scaled_height, scaled_height], align_corners=False)
    distorted_image = tf.squeeze(distorted_image, [0])
    distorted_image = tf.random_crop(distorted_image, [height, width, 3])
    # Restore the shape since the dynamic slice based upon the bbox_size loses the third dimension.
    #distorted_image.set_shape([height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image)
    return distorted_image


def eval_image(image, height, width, scope=None):
    #with tf.op_scope([image, height, width], scope, 'eval_image'):
    with tf.name_scope("eval_image") as scope:
        # Crop the central region of the image with an area containing 87.5% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.95)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
        return image
