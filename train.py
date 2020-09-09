import os
import torch

import parser1
from featureextractor import Model
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

from triplet_loss import TripletLoss, get_dist, get_dist_local
from evaluate import get_acc
from utils import loader, group_imgs


mgpus = False


def save_model(mod, save_path):
    if mgpus:
        torch.save(mod.module.state_dict(), save_path)
    elif mgpus == False:
        torch.save(mod.state_dict(), save_path)


if __name__ == '__main__':

    args = parser1.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader, gallery_loader, query_loader = loader(args)
    ''' load model '''
    print('===> prepare model ...')
    model = Model()
    if mgpus:
        model = torch.nn.DataParallel(model, device_ids=list([0,1])).cuda()
    elif not mgpus:
        model.cuda()  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    t_loss = TripletLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    iters = 0
    best_acc = 0
    best_acc_cos = 0
    long = len(train_loader)
    print_num = int(long/2)
    lr = args.lr

    print('===> start training ...')
    for epoch in range(1, args.epoch + 1):

        model.train()
        avg_loss = 0
        if args.lr_change:
            # Changing learning rate
            lr_diff = ((args.lr*2.5)-args.lr)/args.lr_epochs
            if (epoch < args.lr_epochs+2) & (epoch != 1):
                lr = lr + lr_diff
                print('Changing lr to:', lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr
            else:
                lr = lr * 0.997
                print('Changing lr to:', lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr

        for idx, (imgs, cls) in enumerate(train_loader):
            if args.grouping:
                all_img, all_labels = group_imgs(imgs, cls)
            else:
                all_img, all_labels = imgs.cuda(), cls

            ''' forward path '''
            global_f, local_f, vertical_f, classes = model(all_img)
            global_f, local_f, vertical_f, classes = global_f.cpu(), local_f.cpu(), vertical_f.cpu(), classes.cpu()

            ''' compute loss, back propagation, update parameters '''
            # Global losses
            dist_1_g, dist_2_g = get_dist(global_f, all_labels)
            loss_g = t_loss(dist_1_g, dist_2_g)

            # Local losses
            dist_1_l, dist_2_l = get_dist_local(local_f, all_labels)
            loss_l = t_loss(dist_1_l, dist_2_l)

            # Local Vertical losses (only with model1)
            dist_1_v, dist_2_v = get_dist_local(vertical_f, all_labels)
            loss_v = t_loss(dist_1_v, dist_2_v)

            # Class losses
            loss_c = criterion(classes, all_labels)

            loss = (loss_g * args.global_mult) + (loss_l * args.local_mult) + (loss_c * args.class_mult) + (loss_v * args.vertical_mult)

            model.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters
            avg_loss += loss
            ''' write out information to tensorboard '''
            iters += 1
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)

            if (idx+1) % print_num == 0:
                print('epoch: %d/%d, [iter: %d / %d], Mean loss: %f' \
                      % (epoch, args.epoch, idx + 1, long, avg_loss/print_num))
                avg_loss = 0

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            model.eval()
            acc, acc_c = get_acc(model, query_loader, gallery_loader)
            writer.add_scalar('val_acc_MSE', acc, iters)
            writer.add_scalar('val_acc_Cos', acc_c, iters)
            print('Epoch: [{}] ACC with MSE:{} (max:{})'.format(epoch, acc, best_acc))
            print('Epoch: [{}] ACC with Cosine:{} (max:{})'.format(epoch, acc_c, best_acc_cos))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best_MSE.pth.tar'))
                best_acc = acc

            if acc_c > best_acc_cos:
                save_model(model, os.path.join(args.save_dir, 'model_best_Cos.pth.tar'))
                best_acc_cos = acc_c

    print('best acc for MSE:', best_acc)
    print('best acc for Cos:', best_acc_cos)
