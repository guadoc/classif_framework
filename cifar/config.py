import argparse
import os

def init_config():
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #model config
    parser.add_argument('-mp',       '--model_path'       , default='cifar.models_cifar',                         help='model folder, must contain __init__ file')
    parser.add_argument('-m',        '--model'            , default='inception',                                     help='model file')
    #data config
    parser.add_argument('-dp',       '--dataset'          , default='cifar.dataset_cifar',                        help='folder with metadata configuration')
    parser.add_argument('-d',        '--data'             , default='./cifar/dataset_cifar/bin',                  help='folder with data')
    #saves and logs folders
    parser.add_argument('-ca',       '--cache'            , default='/net/phoenix/blot/cifar/expe',               help='folder with experiments log and variables')
    parser.add_argument('-log',      '--log'              , default='./cifar/expe',                               help='folder with experiments log and variables')
    parser.add_argument('-sm',       '--save_model'       , default=True,                                         help='Decide if model weights are saved after each epoch')
    parser.add_argument('-le',       '--last_epoch'       , default=0,                                            help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=None,                                         help='folder with experiments log and variables, None for last')

    #training config
    parser.add_argument('-ne',       '--n_epoch'          , default=500,                                          help='total number of epoch')
    parser.add_argument('-bs',       '--batch_size'       , default=256,                                          help='number of exemple per batch')
    #parser.add_argument('-chkp',     '--checkpoint'       , default=50,                                          help='number of batch for each checkpoint')
    parser.add_argument('-ntrain',   '--n_data_train'     , default=50000,                                        help='number of data in train set')
    parser.add_argument('-nval',     '--n_data_val'       , default=10000,                                        help='number of data in validation set')
    #not used yet
    parser.add_argument('-lo',       '--loss'             , default='logloss',                                    help='')
    parser.add_argument('-op',       '--optim'            , default='sgd',                                        help='')
    parser.add_argument('-tl',       '--train_loaders'    , default=16,                                           help='')
    parser.add_argument('-vl',       '--val_loaders'      , default=16,                                           help='')
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
