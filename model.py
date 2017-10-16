import tensorflow as tf
import importlib
import os

class Model:
    def __init__(self, opts, data, session=None):
        print('-- Loading model %s'%(opts.model))        
        # creation of a session with memory properties
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        self.sess = session or tf.Session(config=config)
        self.saving_file = os.path.join(opts.cache, opts.model)
        # with tf.device("/gpu:0"):
        model = importlib.import_module(opts.model_path+'.'+opts.model) # models.baseline
        self.inference = model.inference        
        self.layer_regularizer = model.layer_regularizer
        self.optim_param = model.optim_param_schedule or self.default_optim_param_schedule



    def initialize_variables(self, opts):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('variable_to_save'), max_to_keep=100)
        if opts.last_epoch > 0:
            self.model_load(self.saving_file + "_" + str(opts.last_epoch) + ".ckpt")
        print('### Model %s initialized with %d parameters'%(opts.model, self.count_params()))
        #ma_liste = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/conv1/weights')[0]]
        #self.saver = tf.train.Saver(max_to_keep=100)


    def model_save(self, epoch):
        path = self.saving_file + "_" + str(epoch) + ".ckpt"
        save_path = self.saver.save(self.sess, path)
        print("### Model saved in file: %s" % save_path)

    def model_load(self, path):
        print("-- Loading model from file: %s" % path)
        self.saver.restore(self.sess, path)
        print("### Model loaded from file: %s" % path)


    def default_optim_param_schedule(self, monitor):
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
