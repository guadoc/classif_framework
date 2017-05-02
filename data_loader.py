import numpy as np
import threading as th
import tensorflow as tf



class DataLoader:
    def __init__(self, n_loaders, dataset, session):
        def loading(coord, dataset, session):
            while not coord.should_stop():
                batch_sample = dataset.get_next_batch(1)
                session.run(data_fill, feed_dict={data_inputs:batch_sample[0][0], data_labels:batch_sample[1][0]})

        self.n_loader = n_loaders
        self.data_queue = tf.FIFOQueue(100 , [tf.float32, tf.float32], shapes=[dataset.input_shape, dataset.label_shape] )
        data_inputs = tf.placeholder(tf.float32, shape=dataset.input_shape)
        data_labels = tf.placeholder(tf.float32, shape=dataset.label_shape)
        data_fill = self.data_queue.enqueue([data_inputs, data_labels])
        self.loader_coordinator = tf.train.Coordinator()
        self.threads_loader = [th.Thread(target=loading, args =(self.loader_coordinator, dataset, session)) for i in range(self.n_loader)]

        self.quantity = tf.placeholder(tf.int32, shape=[])
        self.data_get = self.data_queue.dequeue_many([self.quantity])
        self.run_options = tf.RunOptions(timeout_in_ms=100)

    def launch(self):
        for t in self.threads_loader: t.start()

    def sample(self, sess, batch_size):
        batch_sample = sess.run(self.data_get, feed_dict={self.quantity:batch_size}, options=self.run_options)
        return batch_sample

    def end(self, sess):
        self.loader_coordinator.request_stop()
        sess.run(self.data_queue.close(cancel_pending_enqueues=True))
