
import tensorflow as tf
import os
import math

class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        self.datapath = opts.data
        if data_type == 'train':
            self.image_number = 50000
        else:
            self.image_number = 10000
        self.image_size = 32
        self.model_size = 28
        self.model_depth = 3
        self.input_shape = [self.model_size, self.model_size, self.model_depth]
        self.input_bat_shape = [None, self.model_size, self.model_size, self.model_depth]
        self.label_bat_shape = [None]
        #self.label_shape = [1]


    def build_input(self, batch_size):
        image_size = 32
        dataset = 'cifar10'
        if dataset == 'cifar10':
            label_bytes = 1
            label_offset = 0
            num_classes = 10
        elif dataset == 'cifar100':
            label_bytes = 1
            label_offset = 1
            num_classes = 100
        else:
            raise ValueError('Not supported dataset %s', dataset)

        depth = 3
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes
        #data_files = tf.gfile.Glob(self.datapath)
        if self.data_type == 'train':
            data_files = [os.path.join(self.datapath, 'data_batch_1.bin'),
            os.path.join(self.datapath, 'data_batch_2.bin'),
            os.path.join(self.datapath, 'data_batch_3.bin'),
            os.path.join(self.datapath, 'data_batch_4.bin'),
            os.path.join(self.datapath, 'data_batch_5.bin')]
        else:
            data_files = [os.path.join(self.datapath, 'test_batch.bin')]


        file_queue = tf.train.string_input_producer(data_files, shuffle=True)
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)

        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        # Convert from string to [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                               [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        if self.data_type == 'train':
            image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
            image = tf.random_crop(image, self.input_shape)
            image = tf.image.random_flip_left_right(image)
            # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
            # image = tf.image.random_brightness(image, max_delta=63. / 255.)
            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.image.per_image_standardization(image)
            example_queue = tf.RandomShuffleQueue(
                capacity= 1000,
                min_after_dequeue= 256,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])
            num_threads = 32
            example_enqueue_op = example_queue.enqueue([image, label])
            tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
                example_queue, [example_enqueue_op] * num_threads))

            # Read 'batch' labels + images from the example queue.
            images, labels = example_queue.dequeue_many(batch_size)
            labels = tf.reshape(labels, [batch_size])
        elif self.data_type == 'val':
            image = tf.image.resize_image_with_crop_or_pad(image, self.input_shape[0], self.input_shape[1])
            image = tf.image.per_image_standardization(image)
            example_queue = tf.FIFOQueue(
                1000,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])
            num_threads = 16
            example_enqueue_op = example_queue.enqueue([image, label])
            tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
                example_queue, [example_enqueue_op] * num_threads))

            # Read 'batch' labels + images from the example queue.
            images, labels = example_queue.dequeue_many(batch_size)
            labels = tf.reshape(labels, [batch_size])
        else:
            print("test data")
            num_threads = 16
            image = tf.image.resize_image_with_crop_or_pad(image, self.input_shape[0], self.input_shape[1])
            image = tf.image.per_image_standardization(image)
            example_queue = tf.FIFOQueue(
                1000,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])
            images = self.test_batch(image)
            labels = label


        #assert len(images.get_shape()) == 4
        #assert tf.equal(images.get_shape().as_list()[0], batch_size)
        #assert images.get_shape()[-1] == self.model_depth
        #assert len(labels.get_shape()) == 1
        #assert labels.get_shape()[0] == batch_size

        #tf.summary.image('images', images)
        return {"inputs":images, "labels":labels}

    def test_batch(self, image):
        image_shape = tf.shape(image)
        batch = []
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
        return batch#tf.Print(batch, ["n crops: ", tf.shape(batch)[0]])
