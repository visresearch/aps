
import math
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torchvision import datasets

from timm.data.mixup import Mixup

from module.model import ViTEncoderPredHead
from module.loss import contrastive_loss

from augmentation import train_transforms,  TwoCropsTransformBox

from utils.misc import AverageMeter
from utils.logger import Logger, console_logger


from config.pretrain.vit_base_pretrain import vit_base_pretrain
from config.pretrain.vit_small_pretrain import vit_small_pretrain
from config.pretrain.vit_tiny_pretrain import vit_tiny_pretrain


def train_epoch(model, optimizer, train_loader, epoch, \
                loggers, args, mixup_fn, scaler):
    model.train()
    
    logger_tb, logger_console = loggers

    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    sims = AverageMeter('Sim', ':.4e')

    num_iter = len(train_loader)
    niter_global = epoch * num_iter

    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        batch_size = images[0][0].size(0)
        image1, idxs1 = images[0]
        image2, idxs2 = images[1]
        
        idxs1 = torch.as_tensor(idxs1, dtype=torch.long)
        idxs1 = idxs1.permute(1, 2, 0)
        idxs2 = torch.as_tensor(idxs2, dtype=torch.long)
        idxs2 = idxs2.permute(1, 2, 0)
        image1 = image1.cuda(args.rank, non_blocking=True)
        image2 = image2.cuda(args.rank, non_blocking=True)

        data_time.update(time.time() - end)

        for j in range(args.nsampling):
            N = image1.shape[0]

            if args.distributed:
                offset = N * torch.distributed.get_rank()   
            else :
                offset=0   
            labels = (torch.arange(N, dtype=torch.long) + offset).cuda() # distribute

            if mixup_fn is not None:
                images1, labels = mixup_fn(images1, labels)

            with autocast():
                idx1 = idxs1[j]
                idx2 = idxs2[j]
                idx1 = idx1.cuda(args.rank)
                idx2 = idx2.cuda(args.rank)
                p1, p2, z1, z2 = model(image1, image2, idx1, idx2)
                loss1, simlarity1 = contrastive_loss(p1, z2.detach(), \
                    args.temp, labels, offset, (mixup_fn is not None), args.distributed)
                loss2, simlarity2 = contrastive_loss(p2, z1.detach(), \
                    args.temp, labels, offset, (mixup_fn is not None), args.distributed)                                    
                loss = loss1 + loss2
                sim = 0.5 * (simlarity1 + simlarity2)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        #-------------------------------------------------------------------------------------------#

        losses.update(loss.item(), batch_size)
        sims.update(sim.item(), batch_size)
        batch_time.update(time.time() - end)

        end = time.time()

        niter_global += 1
        
        if args.rank == 0:
            logger_tb.add_scalar('Iter/loss', losses.val, niter_global)
            logger_tb.add_scalar('Iter/similarity', sims.val, niter_global)
        
        if (i + 1) % args.print_freq == 0 and logger_console is not None \
            and args.rank == 0:  
            lr = optimizer.param_groups[0]['lr']
            logger_console.info(f'Epoch [{epoch}][{i+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {losses.val:.3f}({losses.avg:.3f}),     '
                        f'sim: {sims.val:.3f}({sims.avg:.3f})')

    if args.distributed:
        losses.synchronize_between_processes()
        sims.synchronize_between_processes()

    return losses.avg, sims.avg


def main_worker(gpu, ngpus_per_node, args):

    rank = args.rank * ngpus_per_node + gpu

    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=rank, world_size=args.world_size)
        torch.distributed.barrier()
    args.rank = rank

    #------------------------------logger-----------------------------#
    if args.rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        log_root = args.exp_dir
        name = f'vit_encoder_projection_3layers_with_BN_prediction_'\
            f'3layers_hidden_dim{args.hidden_dim}'    
        logger_tb = Logger(log_root, name)
        logger_console = console_logger(logger_tb.log_dir, 'console')
    else:
        logger_tb,logger_console = None,None
 
    #-----------------------------mixup----------------------------------#
    mixup_fn = None
    mixup_active = False
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.batch_size)

    #---------------------------------model------------------------------#
    if args.arch == 'vit-tiny':
        model = ViTEncoderPredHead(T=args.temp, dim=args.out_dim, \
            mlp_dim=args.hidden_dim, emb_dim=192, num_layer=12, num_head=3)
    elif args.arch == 'vit-small':
        model = ViTEncoderPredHead(T=args.temp, dim=args.out_dim, \
            mlp_dim=args.hidden_dim, emb_dim=384, num_layer=12, num_head=6)
    elif args.arch == 'vit-base':
        model = ViTEncoderPredHead(T=args.temp, dim=args.out_dim, \
            mlp_dim=args.hidden_dim, emb_dim=768, num_layer=12, num_head=12)

    model = model.cuda(args.rank)

    args.lr = args.lr_base * args.batch_size / 256

    if args.distributed :
        torch.cuda.set_device(args.rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size) 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.rank])

    #---------------------------dataload-----------------------#
    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(root=args.data_root, train=True, download=False,
                                 transform=TwoCropsTransformBox(train_transforms, train_transforms, \
                                    args.nsampling, args.sampling_ratio, args.power))
    elif args.dataset == 'cifar100':    
        train_set = datasets.CIFAR100(root=args.data_root, train=True, download=False,
                                 transform=TwoCropsTransformBox(train_transforms, train_transforms, \
                                    args.nsampling, args.sampling_ratio,args.power))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2) 

    #----------------------------optim---------------------------#
    parameters = model.module.parameters() \
        if isinstance(model, DDP) else model.parameters()

    optimizer = torch.optim.AdamW(parameters, 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay)

    lr_func = \
        lambda epoch: min(
            ( epoch + 1) / (args.warmup_epoch + 1e-8), 
              0.5 * (math.cos(epoch / args.nepoch * math.pi) + 1
            )
        )

    scaler = GradScaler()

    start_epoch=0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if args.rank==0 :
        path_save = os.path.join(args.exp_dir, logger_tb.log_name)

    
    for epoch in range(start_epoch,args.nepoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            logger_tb.add_scalar('Epoch/lr', lr, epoch + 1)
        loss, similarity = train_epoch(model, optimizer, train_loader, epoch, (logger_tb, logger_console), args, mixup_fn, scaler)

        lr_scheduler.step()

        if args.rank == 0:
            logger_tb.add_scalar('Epoch/loss', loss, epoch + 1)
            logger_tb.add_scalar('Epoch/similarity', similarity, epoch + 1)

        if (epoch + 1) % args.save_freq == 0 and args.rank == 0:
            _epoch = epoch + 1
            state_dict = model.module.state_dict() \
                if isinstance(model, DDP) else model.state_dict()
            torch.save(state_dict, f'{path_save}/{_epoch:0>4d}.pth') 
    
    if args.rank == 0: 
        state_dict = model.module.state_dict() \
                if isinstance(model, DDP) else model.state_dict()

        torch.save(state_dict, f'{path_save}/last.pth')


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node

    if args.distributed:
        mp.spawn(main_worker,args=(ngpus_per_node, args), nprocs=args.world_size)
    else:
        main_worker(args.rank, ngpus_per_node, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='vit-base', choices=['vit-tiny', 'vit-small', 'vit-base'])
    parser.add_argument("--dataset", type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument("--data-root", type=str, default='./dataset/cifar100')
    parser.add_argument("--nepoch", type=int, default=1600)
    return parser   


if __name__ == '__main__':
    parser = parse_args()
    _args = parser.parse_args()
    if _args.arch == 'vit-tiny':    
        args = vit_tiny_pretrain()
    elif _args.arch == 'vit-small':    
        args = vit_small_pretrain() 
    elif _args.arch == 'vit-base':    
        args = vit_base_pretrain()
    args.dataset = _args.dataset
    args.data_root = _args.data_root
    args.nepoch = _args.nepoch

    main(args)
    