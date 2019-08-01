from dv.DFL import DFL_VGG16, DFL_ResNet50, Energy_ResNet50
from dv.init import *
from dv.MyImageFolderWithPaths import CarsDataset, CUB_2011
from dv.transform import *
from dv.util import *
from train import *
from validate import *
from drawrect import *
import sys
import argparse
import os
import random
import shutil
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='Discriminative Filter Learning within a CNN')
parser.add_argument('--dataroot', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--result', metavar='DIR',
                    help='path to store visualization outputs')
parser.add_argument('--vis_img', metavar='DIR',
                    help='path to get images to visualize')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_filters', default=4, type=int,
                    help='Number of filters per class.')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU nums to use.')
parser.add_argument('--epochs', default=100000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batchsize_per_gpu', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('-testbatch', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--init_type',  default='xavier', type=str,
                    metavar='INIT',help='init net')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--class_idx', default = 27, type = int, metavar='N',
                help='class number of the images to be visualized')
parser.add_argument('--log_train_dir', default='log_train', type=str,
                    help='log for train')
parser.add_argument('--log_test_dir', default='log_test', type=str,
                    help='log for test')
parser.add_argument('--nclass', default=196, type=int,
                    help='num of classes -- stanford cars has 196 classes (default: 196)')
parser.add_argument('--eval_epoch', default=2, type=int,
                    help='every eval_epoch we will evaluate')
parser.add_argument('--vis_epoch', default=2, type=int,
                    help='every vis_epoch we will evaluate')
parser.add_argument('--w', default=448, type=int,
                    help='desired image width to crop, seen as align')
parser.add_argument('--h', default=448, type=int,
                    help='desired image height to crop, seen as align')
parser.add_argument('--dataset',  default='cars', type=str,
                    metavar='dataset',help='Data to use (cars, birds)')

best_prec1 = 0

def main():
    print('Deep Vision <==> Part1 : setting up parameters <==> Begin')
    global args, best_prec1
    args = parser.parse_args()
    print(sys.argv[1:])
    img_dir = os.path.abspath(args.dataroot)
    print('Deep Vision <==> Part1 : setting up parameters <==> Done')

    print('Deep Vision <==> Part2 : loading network  <==> Begin')

    if args.vis_img is not None:
        print('Using VGG Backend...')
        model = DFL_VGG16(k = args.num_filters, nclass = args.nclass) # stanford cars has 196 classes
    else:
        print('Using ResNet Backend...')
        model = DFL_ResNet50(k = args.num_filters, nclass = args.nclass)
        energyNet = Energy_ResNet50(k = args.num_filters, nclass = args.nclass) # for non-random initialization

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model.to(device)
        if args.vis_img is None:
            energyNet.to(device)
        cudnn.benchmark = True

    if args.init_type is not None:
        transform_sample = get_transform_for_test_simple()

        if args.dataset == 'cars':
            sample_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
                                    os.path.join(img_dir,'cars_train'),
                                    os.path.join(img_dir,'devkit/cars_meta.mat'),
                                    transform=transform_sample
                                    )
        elif args.dataset == "birds":
            sample_dataset = CUB_2011(img_dir, train=True, transform=transform_sample, download=True)

        sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = False)

        init_weights(model, init_type=args.init_type) # initialize all layers
        print('Network is initialized with: %s!' % args.init_type)
        if args.vis_img is None:
            center = init_patch(args, sample_loader, energyNet, 1024) #1024 channels in the feature map
            model.state_dict()['conv6.weight'] = center #the 1x1 filters are initialized with patch representations
            print('Patch detectors are initialized with non-random init!')
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Deep Vision <==> Part2 : loading network  <==> Continue from {} epoch {}'.format(args.resume, checkpoint['epoch']))
        else:
            print('Deep Vision <==> Part2 : loading network  <==> Failed')
    print('Deep Vision <==> Part2 : loading network  <==> Done')



    print('Deep Vision <==> Part3 : loading dataset  <==> Begin')

    # transformations are defined in "dv" module
    transform_train = get_transform_for_train()
    transform_test  = get_transform_for_test()
    transform_test_simple = get_transform_for_test_simple()

    if args.dataset == "birds":
        train_dataset = CUB_2011(img_dir, train=True, transform=transform_train)
        test_dataset = CUB_2011(img_dir, train=False, transform=transform_test)
        test_dataset_simple = CUB_2011(img_dir, train=False, transform=transform_test_simple)
    elif args.dataset == "cars":
        train_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
            os.path.join(img_dir,'cars_train'),
            os.path.join(img_dir,'devkit/cars_meta.mat'),
            transform=transform_train
            )
        test_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_test_annos_withlabels.mat'),
            os.path.join(img_dir,'cars_test'),
            os.path.join(img_dir,'devkit/cars_meta.mat'),
            transform=transform_test
            )
        test_dataset_simple = CarsDataset(os.path.join(img_dir,'devkit/cars_test_annos_withlabels.mat'),
            os.path.join(img_dir,'cars_test'),
            os.path.join(img_dir,'devkit/cars_meta.mat'),
            transform=transform_test_simple
            )

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.gpu * args.train_batchsize_per_gpu, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last = False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last = False)
    test_loader_simple = torch.utils.data.DataLoader(
        test_dataset_simple, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last = False)
    print('Deep Vision <==> Part3 : loading dataset  <==> Done')


    print('Deep Vision <==> Part4 : model training  <==> Begin')

    if args.gpu is not None:
        torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, gamma = 0.1)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # check if model is still on GPU
        print('Model on GPU?: ', next(model.parameters()).is_cuda)

        # evaluate on validation set
        if args.evaluate and epoch % args.eval_epoch == 0:
            prec1 = validate_dv(args, test_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'prec1'     : prec1,
            }, is_best)

        # do a test for visualization
        if vis_img is not None and epoch % args.vis_epoch  == 0 and epoch != 0:
            draw_patch_v2(epoch, model, args, args.class_idx)




if __name__ == '__main__':
     main()
