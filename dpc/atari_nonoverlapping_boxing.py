import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../../utils')
sys.path.append('../')
sys.path.append('../..')
from dataset_3d import *
from model_3d import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from dataset_atari import Atari
import gym
sys.path.append('../../d4rl-atari')
import d4rl_atari


# torch.backends.cudnn.benchmark = True


def tensor(X):
    return torch.tensor(X, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='atari', type=str)
parser.add_argument('--seq_len', default=4, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=6, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=1, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=84, type=int)
parser.add_argument('--env_name', default='boxing', type=str)
parser.add_argument('--data_type', default='expert', type=str)
parser.add_argument('--data_seed', default='v0', type=str)
parser.add_argument("--overlapping", action='store_true', help='make frames overlap or not')
parser.add_argument("--load_weights", action='store_true', help='make frames overlap or not')


parser.add_argument('--base_env', default='boxing', type=str)



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    print(f'configuration of the experiment : {args}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # create the model
    model = DPC_RNN(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step)

    model = model.to(cuda)
    if args.load_weights:
        print('======================= Loading pre-trained model ======================')
        weights_ = torch.load(f'../state_dict/{args.base_env}-{args.num_seq}-{args.seq_len}-{args.pred_step}-overlapping={args.overlapping}.pt')
        model.load_state_dict(weights_)
        print('======================= Successfully load model ========================')

    global criterion; criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
    global iteration; iteration = 0


    '''
    No transformation performed on data
    '''
    
    # create the env
    env = gym.make(f'{args.env_name}-{args.data_type}-{args.data_seed}', stack=True)
    dataset = env.get_dataset(n_channels=args.seq_len)

    print('finish create env and dataset')
    # create the dataset
    atari = Atari(dataset=dataset, overlapping=False)
    
    len_atari = len(atari)
    len_train = len_atari * 4 // 5
    len_val = len_atari - len_train

    print(f'length of the dataset : {len_atari}')
    print(f'train / test split : {len_train} / {len_val}')

    atari_train, atari_val = torch.utils.data.random_split(atari, [len_train, len_val])

    # create the dataloader
    train_loader = data.DataLoader(atari_train,
                                   batch_size=args.batch_size,
                                   sampler=data.RandomSampler(atari_train),
                                   shuffle=False,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)

    val_loader = data.DataLoader(atari_val,
                                   batch_size=args.batch_size,
                                   sampler=data.RandomSampler(atari_val),
                                   shuffle=False,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)

    
    
    print('finish create dataloade')

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
    global writer_train
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

#     torch.save(model.state_dict(), f'../state_dict/boxing-resnet18-DPC.pt')
    
    torch.save(model.state_dict(), f'../state_dict/{args.env_name}-{args.num_seq}-{args.seq_len}-{args.pred_step}-overlapping={args.overlapping}.pt')





def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                       nrow=args.num_seq*args.seq_len)),
                                   iteration)
        del input_seq
        
        if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
        # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
        score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
        # print(f'target shape : {target_.shape}')
        # print(B, NP, SQ, B2, NS)
        target_flattened = target_.contiguous().view(B*NP*SQ, B2*NS*SQ)
        # print(target_flattened)
        target_flattened = target_flattened.double().argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

        accuracy_list[0].update(top1.item(),  B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                   epoch, idx, len(data_loader), top1, top3, top5, time.time()-tic, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.contiguous().view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_flattened.double().argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(),  B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
           epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()