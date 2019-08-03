from dv.transform import *
from dv.init import *
from dv.MyImageFolderWithPaths import CarsDataset
from dv.DFL import DFL_ResNet50
from dv.util import *
from train import *
from validate import *
import numpy as np
import sys
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser(description='Discriminative Filter Learning within a CNN')
parser.add_argument('--dataroot', default = './dataset', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--result', default = './vis_result', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU nums to use.')
parser.add_argument('--vis_img', default = './vis_bmw1', metavar='DIR',
                help='path to dataset')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--class_idx', default = 27, type = int, metavar='N',
                help='class number')

def draw_patch_v2(epoch, model, args, class_idx):
    """Draw bounding boxes to locate the discriminative patches

    Usage: images to be visualized are to be put in ./vis_img. They are all
    supposed to be in the same class and the class index is to be passed in.

    Args:
    - epoch: current model epoch that is passed in
    - model: the DFL model
    - class_idx: class index of the images to be visualized.

    Results are saved in the directory specified in --result argument.

    """
    if args.gpu is not None:
        k = model.module.k
    else:
        k = model.k

    result = os.path.abspath(args.result)
    if not os.path.isdir(result):
        os.mkdir(result)

    path_img = os.path.join(os.path.abspath('./'), args.vis_img)
    num_imgs = len(os.listdir(path_img))

    dirs = os.path.join(result, str(epoch))
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    for original in range(num_imgs):
        img_path = os.path.join(path_img, '{}.jpg'.format(original))

        transform1 = get_transform()       # transform for predict
        transform2 = transform_onlysize()  # transform for draw

        img = Image.open(img_path)
        img_pad = transform2(img)
        img_tensor = transform1(img)
        img_tensor = img_tensor.unsqueeze(0)
        out1, out2, out3, indices = model(img_tensor)
        out = out1 + out2 + 0.1 *out3

        value, index = torch.max(out.cpu(), 1)
        vrange = np.arange(0, k)

        for i in vrange:
            indice = indices[0, k*class_idx + i]
            row, col = indice/56, indice%56
            #row, col = indice/28, indice%28 #ResNet feature map size
            p_tl = (8*col, 8*row)
            p_br = (col*8+92, row*8+92)
            draw = ImageDraw.Draw(img_pad)
            draw.rectangle((p_tl, p_br), outline='green')

        # search corresponding classname
        idx = int(index[0]) + 1
        img_dir = os.path.abspath(args.dataroot)
        train_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
                            os.path.join(img_dir,'cars_train'),
                            os.path.join(img_dir,'devkit/cars_meta.mat'),
                            transform=get_transform()
                            )
        #dirname = index2classlist[idx]
        dirname = train_dataset.map_class(idx)
        print("Interpreting for: ", train_dataset.map_class(class_idx), " (true class), ", dirname, " (predicted).")

        filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(original)+'_[predict]_'+str(dirname)+'.jpg'
        filepath = os.path.join(os.path.join(result,str(epoch)),filename)
        img_pad.save(filepath, "JPEG")


def scale_width(img, target_width):
    ow, oh = img.size
    w = target_width
    target_height = int(target_width * oh / ow)
    h = target_height
    return img.resize((w, h), Image.BICUBIC)


def transform_onlysize():
    transform_list = []
    transform_list.append(transforms.Resize(448))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.Pad((42, 42)))
    return transforms.Compose(transform_list)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_transform():
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    return transforms.Compose(transform_list)


def draw_patch(epoch, model, args):
    """Implement: use model to predict images and draw ten boxes by POOL6
    path to images need to predict is in './dataset/bird'

    result : directory to accept images with ten boxes
    subdirectory is epoch, e,g.0,1,2...

    """

    if args.gpu is not None:
        k = model.module.k
    else:
        k = model.k

    result = os.path.abspath(args.result)
    if not os.path.isdir(result):
        os.mkdir(result)

    path_img = os.path.join(os.path.abspath('./'), 'vis_img')
    num_imgs = len(os.listdir(path_img))

    dirs = os.path.join(result, str(epoch))
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    for original in range(num_imgs):
        img_path = os.path.join(path_img, '{}.jpg'.format(original))

        transform1 = get_transform()       # transform for predict
        transform2 = transform_onlysize()  # transform for draw

        img = Image.open(img_path)
        img_pad = transform2(img)
        img_tensor = transform1(img)
        img_tensor = img_tensor.unsqueeze(0)
        out1, out2, out3, indices = model(img_tensor)
        out = out1 + out2 + 0.1 *out3

        value, index = torch.max(out.cpu(), 1)
        vrange = np.arange(0, k)
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9]
        for i in vrange:
            indice = indices[0, i]
            row, col = indice/56, indice%56
            p_tl = (8*col, 8*row)
            p_br = (col*8+92, row*8+92)
            draw = ImageDraw.Draw(img_pad)
            draw.rectangle((p_tl, p_br), outline='red')

        # search corresponding classname
        idx = int(index[0])

        train_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
                            os.path.join(img_dir,'cars_train'),
                            os.path.join(img_dir,'devkit/cars_meta.mat'),
                            transform=get_transform()
                            )
        #dirname = index2classlist[idx]
        dirname = train_dataset.map_class(idx)
        print("Interpreting for: ", train_dataset.map_class(class_idx), " (true class), ", dirname, " (predicted).")

        filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(original)+'_[predict]_'+str(dirname)+'.jpg'
        filepath = os.path.join(os.path.join(result,str(epoch)),filename)
        img_pad.save(filepath, "JPEG")


if __name__ == '__main__':

    args = parser.parse_args()

    print(args)

    model = DFL_ResNet50(k = 4, nclass = 196)
    model = torch.nn.DataParallel(model, device_ids=0)
    resume = 'weight/model_best.pth.tar'
    checkpoint = torch.load(resume, map_location='cpu')
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Deep Vision <==> loading network  <==> Continue from {} epoch {} with acc {}'.format(resume, checkpoint['epoch'], best_prec1))

    draw_patch_v2(args.start_epoch, model, args, args.class_idx)
