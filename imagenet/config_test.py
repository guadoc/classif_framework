import argparse
import os

def init_test_config(model_config):
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp',       '--model_path'       , default='imagenet.models_imagenet',                          help='model folder, must contain __init__ file')
    parser.add_argument('-m',        '--model'            , default='resnet101',                                         help='model file')
    parser.add_argument('-dp',       '--dataset'          , default='imagenet.dataset_imagenet',                         help='folder with metadata configuration')
    #parser.add_argument('-d',        '--data'             , default='/local/common-data/imagenet_2012/images',           help='folder with data')
    #parser.add_argument('-d',        '--data'             , default='/local/chenm/data/imagenet',                        help='folder with data')    
    parser.add_argument('-d',        '--data'             , default='/net/phoenix/blot/imagenet/attacked_data/',                        help='folder with data')    
    parser.add_argument('-le',       '--last_epoch'       , default=1,                                                   help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=1,                                                   help='folder with experiments log and variables, None for last')
    parser.add_argument('-ca',       '--cache'            , default='/net/phoenix/blot/imagenet/expe',                   help='folder with experiments variables')
    parser.add_argument('-log',      '--log'              , default='/net/phoenix/blot/imagenet/expe',                   help='folder with experiments logs')
    #training config
    parser.add_argument('-bs',       '--batch_size'       , default=10,                                                   help='number of exemple per batch')
    parser.add_argument('-nval',     '--n_data_test'       , default=1000,                                               help='number of data in validation set')
    # not in use yet
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
