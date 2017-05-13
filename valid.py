import tensorflow as tf
import tqdm as tq

class Valid:
    def __init__(self, opts, model, dataset):
        self.data = dataset

    def valid(self, monitor, model):
        for bat in tq.tqdm(range(monitor.n_val_batch)):
            batch = self.data.sample(monitor.val_batch_size)
            metrics = model.sess.run(monitor.val_metrics, feed_dict={model.inputs: batch["inputs"], model.labels: batch["labels"], model.training_mode: False})
            monitor.bat_val_update(metrics)
