import tensorflow as tf
import importlib
import os
#from models.baseline import create_model

class Model:
    def __init__(self, opts, data):
        print('-- Loading model %s'%(opts.model))
        model = importlib.import_module(opts.model_path+'.'+opts.model) # models.baseline
        # creation of a session with memory properties
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.saving_file = os.path.join(opts.cache, opts.model)
        self.training_mode = tf.placeholder(tf.bool, shape=[])
        self.inputs = tf.placeholder(tf.float32, shape=data.input_bat_shape)
        self.labels = tf.placeholder(tf.int64, shape=data.label_bat_shape)
        # with tf.device("/gpu:0"):
        self.outputs, self.infos = model.inference(self.inputs, self.training_mode)
        #self.regularizer = tf.reduce_sum(model.regularizer())
        self.layer_regularizer = model.layer_regularizer()
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels))
        self.optim_param = model.optim_param_schedule or self.optim_param_schedule
        self.saver = tf.train.Saver(max_to_keep=100)


    def initialize_variables(self, opts):
        if opts.last_epoch > 0:
            self.sess.run(tf.global_variables_initializer())
            self.model_load(os.path.join(opts.cache, opts.model + "_" + str(opts.last_epoch) + ".ckpt"))
        else:
            self.sess.run(tf.global_variables_initializer())
        print('### Model %s initialized with %d parameters'%(opts.model, self.count_params()))


    def set_train_mode(self, training_mode = True):
        _ = self.sess.run([self.training_mode], feed_dict={self.training_mode: training_mode})


    def model_save(self, epoch):
        path = self.saving_file + "_" + str(epoch) + ".ckpt"
        save_path = self.saver.save(self.sess, path)
        print("### Model saved in file: %s" % save_path)

    def model_load(self, path):
        print("-- Loading model from file: %s" % path)
        self.saver.restore(self.sess, path)
        print("### Model loaded from file: %s" % path)


    def optim_param_schedule(self, monitor):
        return {"lr": 0.01, 'momentum': 0.9}

    def count_params(self):
        tot_nb_params = 0
        def get_nb_params_shape(shape):
            nb_params = 1
            for dim in shape:
                nb_params = nb_params*int(dim)
            return nb_params
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params
