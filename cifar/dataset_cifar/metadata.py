
import tensorflow as tf
import os
import math

class Metadata:
    def __init__(self, data_type, opts):
        self.data_type = data_type
        self.datapath = opts.data
        if data_type == 'train':
            self.image_number = opts.n_data_train
        else:
            self.image_number = opts.n_data_val
        self.image_size = 36#32#36
        self.model_size = 32#28#32
        self.model_depth = 3
        self.input_shape = [self.model_size, self.model_size, self.model_depth]
        self.image_shape = [self.image_size, self.image_size, self.model_depth]
        self.input_bat_shape = [None, self.model_size, self.model_size, self.model_depth]
        self.label_bat_shape = [None]
        self.ntrain_loader = opts.train_loaders
        self.nval_loader = opts.val_loaders


    def build_input(self, batch_size):
        image_size = 32
        depth = 3
        dataset = 'cifar100'
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

        if self.data_type == 'train':
            shuffle = True
        else:
            shuffle = False

        file_queue = tf.train.string_input_producer(data_files, shuffle=shuffle)
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)
        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        # Convert from string to [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]), [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        if self.data_type == 'train':
            num_threads = self.ntrain_loader
            #image = tf.image.resize_image_with_crop_or_pa(image, self.input_shape[0], self.input_shape[1])
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size, self.image_size)
            image = tf.random_crop(image, self.input_shape)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.per_image_standardization(image)
            example_queue = tf.RandomShuffleQueue(
                capacity= 1000,
                min_after_dequeue= 256,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])

        elif self.data_type == 'val':
            num_threads = self.nval_loader
            image = tf.image.resize_image_with_crop_or_pad(image, self.input_shape[0], self.input_shape[1])
            image = tf.image.per_image_standardization(image)
            example_queue = tf.FIFOQueue(
                1000,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])
        else:
            num_threads = self.nval_loader
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size, self.image_size)
            image = tf.image.per_image_standardization(image)
            example_queue = tf.FIFOQueue(
                1000,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.image_shape, [1]])


        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))
        #images, labels = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, enqueue_many=False, capacity=2000, allow_smaller_final_batch=True)
        #example_queue = tf.Print(example_queue, [example_queue.size()])
        images, labels = example_queue.dequeue_many(batch_size)
        labels = tf.reshape(labels, [batch_size])

        if self.data_type == 'test':
            images = self.inputs_augment(images, batch_size)

        #assert len(images.get_shape()) == 4
        #assert tf.equal(images.get_shape().as_list()[0], batch_size)
        #assert images.get_shape()[-1] == self.model_depth
        #assert len(labels.get_shape()) == 1
        #assert labels.get_shape()[0] == batch_size
        #tf.summary.image('images', images)

        return {"inputs":images, "labels":labels}

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
