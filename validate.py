import torch
import time
import sys
from dv.util import *
from dv.save import *
from tqdm import tqdm

def validate_dv(args, val_loader, model, criterion, epoch):
    print('Deep Vision module: validating on the test set...')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()
    end = time.time()

    total_output= []
    total_label = []
    start_test = True

    # we may have ten d in data
    for i, (data, target) in tqdm(enumerate(val_loader)):
        target = target.type(torch.LongTensor)

        if args.gpu is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device).reshape(data.size(0) * 10, 3, 448, 448) # bs*10, 3, 448, 448
            target = target.to(device)
            target = target.resize(int(data.size(0)/10), 1).expand(int(data.size(0)/10),10).resize(data.size(0))

        output1, output2, output3, _ = model(data)
        output = output1 + output2 + 0.1 * output3

        if start_test:
           total_output = output.data.float()
           total_label = target.data.float()
           start_test = False
        else:
           total_output = torch.cat((total_output, output.data.float()) , 0)
           total_label = torch.cat((total_label , target.data.float()) , 0)

    _,predict = torch.max(total_output,1)

    acc = torch.sum(torch.squeeze(predict).float() == total_label).item() / float(total_label.size()[0])
    print(' test acc == ' + str(acc))
    log.save_test_info(epoch, top1=acc, top5=None)
    return acc


def validate(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()
    end = time.time()

    # we may have ten d in data
    for i, (data, target, paths) in enumerate(val_loader):
        if args.gpu is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            target = target.to(device)

        # compute output
        for idx, d in enumerate(data[0]):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]
            output1, output2, output3, _ = model(d)
            output = output1 + output2 + 0.1 * output3

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1.update(prec1[0], 1)
            top5.update(prec5[0], 1)
            print('DFL-CNN <==> Test <==> Img:{} No:{} Top1 {:.3f} Top5 {:.3f}'.format(i, idx, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))

    print('Deep Vision <==> Test Total <==> Top1 {:.3f}% Top5 {:.3f}%'.format(top1.avg, top5.avg))
    log.save_test_info(epoch, top1, top5)
    return top1.avg


def validate_simple(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()
    end = time.time()

    # we may have ten d in data
    for i, (data, target) in enumerate(val_loader):
        target = target.type(torch.LongTensor)

        if args.gpu is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            target = target.to(device)

        # compute output
        for idx, d in enumerate(data):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]
            output1, output2, output3, _ = model(d)
            output = output1 + output2 + 0.1 * output3

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1.update(prec1[0], 1)
            top5.update(prec5[0], 1)
            if i % 1000 == 0:
                print('DFL-CNN <==> Test <==> Img:{} Top1 {:.3f} Top5 {:.3f}'.format(i, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))

    print('Deep Vision <==> Test Total <==> Top1 {:.3f}% Top5 {:.3f}%'.format(top1.avg, top5.avg))
    log.save_test_info(epoch, top1, top5)
    return top1.avg
