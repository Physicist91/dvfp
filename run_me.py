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
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm

settings = {'use_gpu':True,
            'num_epochs':2,
    'num_workers':4,
    'batch_size':16,
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
        cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=mom, weight_decay = wdecay)
    lr_policy = lr_scheduler.StepLR(optimizer, step_size=lr_updates, gamma=lr_gamma)

    # Optionally resume from a checkpoint
    if settings['resume']:

        if os.path.isfile(settings['resume']):
            # load stuffs
            checkpoint = torch.load(settings['resume'])
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loading Network  <==> Continue from {} at epoch #{}'.format(settings['resume'], checkpoint['epoch']))

        else:
            print('Loading Network  <==> Failed')
    else:
        best_top1 = 0

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

    train_loader = DataLoader(train_dataset, batch_size=m, shuffle=True, num_workers=nworkers, pin_memory=True)
    print("train size:", len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=m, shuffle=True, num_workers=nworkers, pin_memory=True)
    test_loader_noTC = DataLoader(test_dataset_noTC, batch_size=m, shuffle=True, num_workers=nworkers, pin_memory=True)
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
        losses = np.zeros(5) # to store individual losses from the branches

        for c, data in tqdm(enumerate(train_loader)):
            inputs, labels = data
            labels = labels.type(torch.LongTensor)

            if settings['use_gpu']:
                 inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            out_g, out_p, out_s, _ = net(inputs)
            
            out = out_g + out_p + 0.1 * out_s # the weights for each stream as in the paper

            loss_g = criterion(out_g, labels)
            loss_p = criterion(out_p, labels)
            loss_s = criterion(out_s, labels)
            
            loss = loss_g + loss_p + 0.1 * loss_s
            
            top1, top5 = dv.helpers.get_accuracy(out, labels, topk=(1, 5))  # paper cited only top-1
            
            _, predicted = torch.max(out.data, 1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            assert not np.isnan(running_loss), "Fuck, loss blows up!"
            
            losses[0] += loss_g
            losses[1] += loss_p
            losses[2] += loss_s
            losses[3] += top1[0]
            losses[4] += top5[0]
            
            # backward pass + optimize
            loss.backward()
            optimizer.step()
            
            if (c+1) % 40 == 0:
                losses1 = losses/c
                
                print('Train epoch: {} [{}/{}] || Loss: G-Stream = {:.4f}, P-Stream = {:.4f}, side = {:.4f} ||\n'
                'Top-1 Acc: {:.2f} || Top-5 Acc: {:.2f}'.format(
                epoch, c, len(train_loader), losses1[0], losses1[1], losses1[2], losses1[3], losses1[4]))
                
                helpers.save_train_info(epoch, c, len(train_loader), running_loss/c, losses1)
                
        epoch_loss = running_loss / c
        train_acc = 100 * correct / total
        print('Summary of epoch: {} || Loss: {:.4f} || Acc: {:.2f} %%'.format(epoch, epoch_loss, train_acc))
    
    # model evaluation step: to be combined into training later
    net.eval()
    correct = 0
    total = 0
    for c, data in tqdm(enumerate(test_loader_noTC)):
        images, labels = data
        labels = labels.type(torch.LongTensor)
        if settings['use_gpu']:
            labels = labels.to(device)
            images = images.to(device)
        
        for idx, img in enumerate(images): # [batchsize, 10_crop, 3, 448, 448]
            img = img.unsqueeze(0) # img [1, 3, 448, 448]
            out_g, out_p, out_s, _ = net(img)
            out = out_g + out_p + 0.1 * out_s
            top1, top5 = dv.helpers.get_accuracy(out, labels, topk=(1, 5))  # paper cited only top-1
            correct += top1
            total += 1
    print('Validation Acc: {:.2f} %%'.format(correct/total))
    
    # saving the model
    if top1 > best_top1:
        best_top1 = top1
    helpers.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_top1': best_top1,
                'optimizer' : optimizer.state_dict(),
                'top1'     : correct/total,
            }, is_best = top1 > best_top1)
    
    
    #TODO: visualization

    print("Success, Motherfucker!")
