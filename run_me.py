# begin by importing our "Deep Vision" module (or dv in short)
import dv
from dv.model import DeepVision_VGG16
from dv.ImageFolder import CarsDataset
import dv.helpers as helpers
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm

settings = {'use_gpu':True,
            'num_epochs':2,
    'num_workers':4,
    'batch_size':4,
    'lr0':0.1, #initial learning rate
    'lr_updates':10, # stepsize frequency of learning rate decay
    'lr_gamma':0.1, #how much to drop the learning rate
    'momentum':0.9,
    'weight_decay':0.000005,
    'resume':False, #path to model checkpoint
    'data_dir':'/export/home/dv/dv029/workspace/data'
}

# Paper use input size of 448 x 448, we will use random crop to this size
def transform_train():
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x:helpers.rescale(x, 448)))
    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    transform_list.append(transforms.RandomCrop((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    return transforms.Compose(transform_list)

def transform_test():
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x:helpers.rescale(x, 560)))
    transform_list.append(transforms.TenCrop(448))
    transform_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))((transforms.ToTensor())(crop)) for crop in crops])) )
    return transforms.Compose(transform_list)

def transform_test_noTC():
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x:helpers.rescale(x, 448)))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    return transforms.Compose(transform_list)

if __name__ == '__main__':

    print('Deep Vision (+:===:+) PART 1 : setup (+:===:+) (*^o^*) Begin')

    lr0 = settings['lr0']
    lr_updates = settings['lr_updates']
    lr_gamma = settings['lr_gamma']
    mom = settings['momentum']
    wdecay = settings['weight_decay']
    resume = settings['resume']
    m = settings['batch_size']
    img_dir = settings['data_dir']
    nworkers = settings['num_workers']
    num_epochs = settings['num_epochs']

    net = DeepVision_VGG16(k = 10, M = 200)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if settings['use_gpu']:
        print("Using ", torch.cuda.device_count(), " GPUs...")
        net = torch.nn.DataParallel(net)
        net = net.to(device)
        #cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=mom, weight_decay = wdecay)
    lr_policy = lr_scheduler.StepLR(optimizer, step_size=lr_updates, gamma=lr_gamma)

    # Optionally resume from a checkpoint
    if resume:

        if os.path.isfile(resume):
            # load stuffs
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loading Network  <==> Continue from {} at epoch #{}'.format(resume, checkpoint['epoch']))

        else:
            print('Loading Network  <==> Failed')

    print('Deep Vision (+:===:+) PART 1 : setup (+:===:+) *\(^o^)/* End')

    ####################################################################
    ####################################################################
    print('Deep Vision (+:===:+) PART 2 : loading dataset (+:===:+) (*^o^*) Begin')

    train_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_train_annos.mat'),
                            os.path.join(img_dir,'cars_train'),
                            os.path.join(img_dir,'devkit/cars_meta.mat'),
                            transform=transform_train()
                            )

    test_dataset = CarsDataset(os.path.join(img_dir,'devkit/cars_test_annos_withlabels.mat'),
                            os.path.join(img_dir,'cars_test'),
                            os.path.join(img_dir,'devkit/cars_meta.mat'),
                            transform=transform_test()
                            )

    test_dataset_noTC = CarsDataset(os.path.join(img_dir,'devkit/cars_test_annos_withlabels.mat'),
                            os.path.join(img_dir,'cars_test'),
                            os.path.join(img_dir,'devkit/cars_meta.mat'),
                            transform=transform_test_noTC()
                            )

    train_loader = DataLoader(train_dataset, batch_size=m,
                            shuffle=True, num_workers=nworkers)
    print("train size:", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=m,
                            shuffle=True, num_workers=nworkers)
    test_loader_noTC = DataLoader(test_dataset_noTC, batch_size=m,
                            shuffle=True, num_workers=nworkers)
    print("test size without tencrop:", len(test_dataset_noTC))
    print("test size:", len(test_dataset))

    print('Deep Vision (+:===:+) PART 2 : loading dataset (+:===:+) *\(^o^)/* End')
    #######################################################
    #######################################################

    print('Deep Vision (+:===:+) PART 3 : model training (+:===:+) (*^o^*) Begin')

    for epoch in range(num_epochs):
        lr_policy.step()
        net.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for c, data in tqdm(enumerate(train_loader)):
            inputs, labels = data
            labels = labels.type(torch.LongTensor)

            if settings['use_gpu']:
                 inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            else:
                 inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            out_g, out_p, out_s, _ = net(inputs)

    print("Success, Motherfucker!")
