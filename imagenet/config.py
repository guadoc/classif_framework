import argparse
import os

def init_config():
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp',       '--model_path'       , default='imagenet.models_imagenet',                                      help='model folder, must contain __init__ file')
    parser.add_argument('-m',        '--model'            , default='resnet',                                     help='model file')

    parser.add_argument('-dp',       '--dataset'          , default='imagenet.dataset_imagenet',                         help='folder with metadata configuration')
    parser.add_argument('-d',        '--data'             , default='/local/common-data/imagenet_2012/images',     help='folder with data')
    #parser.add_argument('-d',        '--data'             , default='./dataset_cifar',     help='folder with data')

    parser.add_argument('-ca',       '--cache'            , default='/net/phoenix/blot/imagenet/expe',                      help='folder with experiments log and variables')
    parser.add_argument('-log',       '--log'             , default='/net/phoenix/blot/imagenet/expe',                      help='folder with experiments log and variables')
    parser.add_argument('-save',     '--saving_model'     , default=True,                                         help='Decide if model weights are saved after each epoch')
    parser.add_argument('-le',       '--last_epoch'       , default=0,                                             help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=None,                                             help='folder with experiments log and variables, None for last')

    parser.add_argument('-ne',       '--n_epoch'          , default=120,                                           help='total number of epoch')
    parser.add_argument('-bs',       '--batch_size'       , default=48,                                           help='number of exemple per batch')
    #parser.add_argument('-chkp',     '--checkpoint'       , default=10,                                            help='number of batch for each checkpoint')
    parser.add_argument('-ntrain',   '--n_data_train'     , default=1280000,                                       help='number of data in train set')
    parser.add_argument('-nval',     '--n_data_val'       , default=50000,                                         help='number of data in validation set')
    # not in use yet
    parser.add_argument('-op',       '--optim'            , default='sgd',                                             help='number of thread feeding queue during training')
    parser.add_argument('-tl',       '--train_loaders'    , default=16,                                             help='number of thread feeding queue during training')
    parser.add_argument('-vl',       '--val_loaders'      , default=16,                                             help='number of thread feeding queue during validation')
    args = parser.parse_args()

    # take the last index and add 1 to it in the name of the experimental directory
    args.index = args.index or index_max(args.log)
    print("### Experience indexed: %d"%(args.index))
    args.cache = os.path.join(args.cache, str(args.index) + "_" + args.model)
    args.log = os.path.join(args.log, str(args.index) + "_" + args.model)
    try:
        os.stat(args.cache)
    except:
        os.mkdir(args.cache)
    try:
        os.stat(args.log)
    except:
        os.mkdir(args.log)
    print('### Logs path is ' + args.log)
    print('### Cache path is ' + args.cache)

    return args

def index_max(path):
    index = 0
    for f in os.listdir(path):
        if not os.path.isfile(f):
            tab = f.split("_")
            ind = int(float(tab[0]))
            if ind > index:
                index=ind
    return index+1
