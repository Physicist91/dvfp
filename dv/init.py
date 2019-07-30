import torch
from torch.nn import init
from torch.autograd import Variable
from tqdm import tqdm

def init_patch(args, val_loader, model, n_channels):
    """
    Initialization for the patch detectors
    """
    print('Initializing patch detectors...')
    model.eval()
    labels_set = set()

    for batches in tqdm(val_loader):
        if len(labels_set) >= model.nclass:
            break

        data, target = batches
        if target.item() in labels_set:
            continue
        else:
            labels_set.add(target.item())
            idx = target.item()

            if args.gpu is not None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data = Variable(data.to(device))
                target = Variable(target.to(device))

            result = torch.zeros(n_channels, model.k * model.nclass)
            for j, d in enumerate(data):  # data [batchsize, 3, 448, 448]
                d = d.unsqueeze(0) # d [1, 3, 448, 448]
                center = model(d)
                result[:, idx*model.k : idx*model.k + model.k] = center

    return result.view(-1, n_channels)

def init_net(net, init_type='normal'):
    init_weights(net, init_type)
    return net

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
