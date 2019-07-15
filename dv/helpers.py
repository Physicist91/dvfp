import os
from PIL import Image

def rescale(img, fixed_min):
    ow, oh = img.size
    if ow < oh:      
        nw = fixed_min
        nh = nw * oh // ow   
    else:      
        nh = fixed_min 
        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)


def get_accuracy(output, target, topk=(1, 5)):
    
    """ Compute accuracy for Top-k
    This function is only for computing accuracies per batch
    
    Args:
    * Output
    * Target
    * Tuple of Top-K accuracies desired
    
    Returns: a list of accuracies corresponding to top-k
    """
    
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t()            
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """save model weights
    
    [description]
    
    Arguments:
        state {[type]} -- [description] a dict describe some params
        is_best {bool} -- [description] a bool value
    
    Keyword Arguments:
        filename {str} -- [description] (default: {'checkpoint.pth.tar'})
    """
    root_dir = get_root_dir()
    weight_dir = os.path.join(root_dir, 'weight')
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    
    epoch = state['epoch']
    prec1 = state['top1']

    file_path = os.path.join(weight_dir, 'epoch_{:04d}_top1_{:02d}_{}'.format(int(epoch), int(prec1), filename))  
    torch.save(state, file_path)
    
    best_path = os.path.join(weight_dir, 'model_best.pth.tar')
    
    if is_best:
        shutil.copyfile(file_path, best_path)

        
def save_train_info(epoch, batch, maxbatch, losses):
        """
        Helper function to save training information
        """
        loss = losses[0]
        loss1 = losses[1]
        loss2 = losses[2]
        loss3 = losses[3]
        root_dir = os.path.abspath('./')
        log_dir = os.path.join(root_dir, 'log') 
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        log_file = os.path.join(log_dir, 'log_train.txt')
        #if not os.path.exists(log_file):
        #    os.mknod(log_file)

        with open(log_file, 'w') as f:
            f.write('Epoch: [{}][{}/{}]\n'
                    'Loss {:.4f} \t'
                    'Loss G-Stream {:.4f} \t'
                    'Loss P-Stream {:.4f} \t'
                    'Loss Side Branch {:.4f} \n'
                    'Top-1 accuracy ({:.3f})\t'
                    'Top-5 accuracy ({:.3f})\n'.format(epoch, batch, maxbatch, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
            
            