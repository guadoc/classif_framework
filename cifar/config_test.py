import argparse
import os

def init_test_config(model_config):
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #model config
    parser.add_argument('-mp',       '--model_path'       , default='cifar.models_cifar',                         help='model folder, must contain __init__ file')
    #data config
    parser.add_argument('-m',        '--model'            , default='inception',                                        help='model file')
    parser.add_argument('-dp',       '--dataset'          , default='cifar.dataset_cifar',                        help='folder with metadata configuration')
    parser.add_argument('-d',        '--data'             , default='./cifar/dataset_cifar/bin',                  help='folder with data')
    #saves and logs folders
    parser.add_argument('-le',       '--last_epoch'       , default=0,                                            help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=0,                                         help='folder with experiments log and variables, None for last')
    parser.add_argument('-ca',       '--cache'            , default='/net/phoenix/blot/cifar/expe',               help='folder with experiments variables')
    #test config
    parser.add_argument('-bs',       '--batch_size'       , default=40,                                          help='number of exemple per batch')
    parser.add_argument('-ntest',    '--n_data_test'      , default=10000,                                        help='number of data in validation set')

    args = parser.parse_args()
    # take the last index and add 1 to it in the name of the experimental directory
    args.index = model_config[0] or args.index
    args.last_epoch = model_config[1] or args.last_epoch
    print("### Model from index: %d"%(args.index))
    folder_cache = os.path.join(args.cache, str(args.index) + "_" + args.model)
    try:
        os.stat(folder_cache)
    except:
        print("no cache at "+folder_cache)
    args.cache = os.path.join(folder_cache, "models")
    try:
        os.stat(args.cache)
    except:
        os.mkdir(args.cache)
    print('### Cache path is ' + args.cache)
    return args
