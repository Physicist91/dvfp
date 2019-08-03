from dv.transform import *
from dv.init import *
from dv.MyImageFolderWithPaths import CarsDataset
from dv.DFL import DFL_ResNet50
from dv.util import *
from train import *
from validate import *
import sys
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFont, ImageDraw
import os
import re
import numpy as np

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

def read_specific_line(line, path):
    target = int(line)
    with open(path, 'r') as f:
        line = f.readline()
        c = []
        while line:
            currentline = line
            c.append(currentline)
            line = f.readline()

    reg =  c[target-1].split(',')[-1]
    return reg

def path_to_contents(path):
    filename = path.split('/')[-1]
    index_gtline = re.split('_|.jpg', filename)[-2]
    index_image = filename.split('_')[1]
    gt_dir = '/data1/data_sdj/ICDAR2015/end2end/train/gt'
    gt_file = os.path.join(gt_dir, 'gt_img_'+str(index_image)+'.txt')
    # I want to read gt_file of specific line index_gtline
    contents = read_specific_line(int(index_gtline), gt_file)
    #print(index_image, index_gtline, contents)
    return contents

def create_font(fontfile, contents):
    # text and font
    unicode_text = contents
    if isinstance(unicode_text,str) and unicode_text.find('###') != -1 or unicode_text == '':
        print('######################')
        return None
    try:
        font = ImageFont.truetype(fontfile, 36, encoding = 'unic')

        # get line size
        # text_width, text_font.getsize(unicode_text)

        canvas = Image.new('RGB', (128, 48), "white")

        draw = ImageDraw.Draw(canvas)
        draw.text((5,5), unicode_text, 'black', font)

    #canvas.save('unicode-text.png','PNG')
    #canvas.show()
        print(canvas.size)
        return canvas
    except:
        return None

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    #imga = Image.fromarray(imga)
    #imgb = Image.fromarray(imgb)
    w1,h1 = imga.size
    w2,h2 = imgb.size
    img = Image.new("RGB",(256, 48))
    img.paste(imga, (0,0))
    img.paste(imgb, (128, 0))
    return img

def get_transform():
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))
    #transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    return transforms.Compose(transform_list)

def draw_patch_v2(epoch, model, args):
    """Implement: use model to predict images and draw ten boxes by POOL6
    path to images need to predict is in './vis_img'

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

    path_img = os.path.join(os.path.abspath('./'), args.vis_img)

    num_imgs = list(os.listdir(path_img))

    dirs = os.path.join(result, str(epoch))
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    
    img_dir = os.path.abspath(args.dataroot)
    train_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
                        os.path.join(img_dir,'cars_train'),
                        os.path.join(img_dir,'devkit/cars_meta.mat'),
                        transform=get_transform()
                        )
    classnames = len(num_imgs)*[0]
    for el in train_dataset.car_annotations:
        if el[-1][0] in num_imgs:
            classnames[num_imgs.index(el[-1][0])] = el[-2][0][0]

    for j, original in enumerate(num_imgs):
        img_path = os.path.join(path_img, '{}'.format(original))

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
            indice = indices[0, k*classnames[j] + i]
            row, col = indice/56, indice%56
            #row, col = indice/28, indice%28 #ResNet feature map size
            p_tl = (8*col, 8*row)
            p_br = (col*8+92, row*8+92)
            draw = ImageDraw.Draw(img_pad)
            draw.rectangle((p_tl, p_br), outline='green')

        # search corresponding classname
     
        #dirname = index2classlist[idx]
        dirname = train_dataset.map_class(int(index)+1)
        print("Interpreting for: ", train_dataset.map_class(classnames[j]), " (true class), ", dirname, " (predicted).")

        filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(original)+'_[predict]_'+str(dirname)+'.jpg'
        filepath = os.path.join(os.path.join(result,str(epoch)),filename)
        img_pad.save(filepath, "JPEG")

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
