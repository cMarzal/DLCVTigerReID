from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Parser for TigerReID by team Wondaba for course DLCV')

    '''Enviroment parameter'''
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--random_seed', type=int, default=836)

    '''Datasets parameters'''
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help="root path to data directory")
    parser.add_argument('--workers', default=1, type=int,
                       help="number of data loading workers (default: 4)")

    '''Training parameters'''
    parser.add_argument('--epoch', default=500, type=int,
                        help="num of total epochs")
    parser.add_argument('--train_batch', default=10, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=40, type=int,
                        help="query batch size")
    parser.add_argument('--val_epoch', default=4, type=int,
                        help="num of epochs a val is run")
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--label_group', default=3, type=int,
                        help="Number of grouped images with the same label")

    '''Losses weight multiplyers'''
    parser.add_argument('--global_mult', default=3, type=float,
                        help="Multiplyer of global losses")
    parser.add_argument('--local_mult', default=0.5, type=float,
                        help="Multiplyer of local losses")
    parser.add_argument('--class_mult', default=1.5, type=float,
                        help="Multiplyer of class losses")
    parser.add_argument('--vertical_mult', default=0.1, type=float,
                        help="Multiplyer of vertical losses")


    '''Optimizer params'''
    parser.add_argument('--lr', default=0.00009, type=float,
                       help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help="initial Weight Decay")
    parser.add_argument('--lr_epochs', default=100, type=float,
                        help="initial learning rate")

    '''Optional Improvements'''
    parser.add_argument('--lr_change', default=1, type=bool,
                        help="Change learning rate or not")
    parser.add_argument('--grouping', default=True, type=bool,
                        help="Groups the images in same categories or not")
    parser.add_argument('--random_sampling', default=True, type=bool,
                        help="Change the batches of images randomly(increase epoch, and only when grouping = true)")

    '''Final Script parameters'''
    parser.add_argument('--img_dir', type=str, default='dataset/imgs',
                        help="root path to data directory")
    parser.add_argument('--query_dir', type=str, default='dataset/query2.csv',
                        help="root path to data directory")
    parser.add_argument('--gallery_dir', type=str, default='dataset/gallery2.csv',
                        help="root path to data directory")
    parser.add_argument('--csv_dir', type=str, default='predict.csv',
                        help="root path to data directory")

    args = parser.parse_args()

    return args
