import torch
import os
import shutil
import datetime
from dv.util import *

class Log(object):
    def save_train_info(self, epoch, batch, maxbatch, losses, top1, top5):
        """
        loss may contain several parts
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
        log_file_minimal = os.path.join(log_dir, 'log_train_minimal.txt')
        if not os.path.exists(log_file):
            os.mknod(log_file)
        if not os.path.exists(log_file_minimal):
            os.mknod(log_file_minimal)
            with open(log_file_minimal, 'a') as g:
                g.write('epoch\tloss\tloss_avg\tloss1\tloss1_avg\tloss2\tloss2_avg\tloss3\tloss3_avg\tprec1\tprec5\n')

        with open(log_file, 'a') as f:
            f.write('DFL-CNN <==> Train <==> Epoch: [{0}][{1}/{2}]\n'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                    'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\n'
                    'Prec@1 ({top1.avg:.3f})\t'
                    'Prec@5 ({top5.avg:.3f})\n'.format(epoch, batch, maxbatch,loss = loss,loss1 = loss1,loss2 = loss2, loss3=loss3, top1=top1, top5=top5))

        with open(log_file_minimal, 'a') as g:
            g.write('{0}\t'
                    '{loss.val:.4f}\t{loss.avg:.4f}\t'
                    '{loss1.val:.4f}\t{loss1.avg:.4f}\t'
                    '{loss2.val:.4f}\t{loss2.avg:.4f}\t'
                    '{loss3.val:.4f}\t{loss3.avg:.4f}\t'
                    '{top1.avg:.3f}\t'
                    '{top5.avg:.3f}\n'.format(epoch, loss=loss, loss1=loss1, loss2=loss2, loss3=loss3, top1=top1, top5=top5))


    def save_test_info(self, epoch, top1, top5):
        root_dir = os.path.abspath('./')
        log_dir = os.path.join(root_dir, 'log')
        # check log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(log_file_minimal):
            os.mknod(log_file_minimal)
        log_file = os.path.join(log_dir, 'log_test.txt')
        log_file_minimal = os.path.join(log_dir, 'log_test_minimal.txt')
        if top5 == None:
            if not os.path.exists(log_file_minimal):
                os.mknod(log_file_minimal)
                with open(log_file_minimal, 'a') as g:
                    g.write('epoch\taccuracy\n')
            with open(log_file_minimal, 'a') as g:
                g.write('{0}\t{acc:.6f}\n'.format(epoch, acc=top1))
            return


        if not os.path.exists(log_file):
            os.mknod(log_file)
        if not os.path.exists(log_file_minimal):
            os.mknod(log_file_minimal)
            with open(log_file_minimal, 'a') as g:
                g.write('epoch\ttop1\ttop5\n')

        with open(log_file, 'a') as f:
            f.write('DFL-CNN <==> Test <==> Epoch: [{0}] Top1:{top1.avg:.3f}% Top5:{top5.avg:.3f}%\n'.format(epoch, top1=top1, top5=top5))
	
        with open(log_file_minimal, 'a') as g:
            g.write('{0}\t{top1.avg:.3f}\t{top5.avg:.3f}\n'.format(epoch, top1=top1, top5=top5))

# this is for weight
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """[summary]

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
    prec1 = state['prec1']

    file_path = os.path.join(weight_dir, 'epoch_{:04d}_top1_{:02d}_{}'.format(int(epoch), int(prec1), filename))
    torch.save(state, file_path)

    best_path = os.path.join(weight_dir, 'model_best.pth.tar')

    if is_best:
        shutil.copyfile(file_path, best_path)
