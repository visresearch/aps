import argparse


def vit_base_finetune():
    args = argparse.Namespace()

    args.dataset = 'cifar10'
    args.data_root = '/path/to/dataset/'
    args.arch = 'vit-base'
    args.pretrained_weights = './weight/pretrain/tiny_1600ep_1e-3_100_0.15.pth'
    args.input_size = 32
    args.epochs = 100
    args.start_epoch = 0
    args.num_workers = 10
    args.pin_mem = True
    args.batch_size = 128
    args.output_dir = './out'
    args.seed = 0

    # ---ema----------
    args.model_ema = True
    args.model_ema_decay = 0.99996
    args.model_ema_force_cpu = False 
    args.drop_path = 0.1


    # Optimizer parameters
    args.opt = 'adamw'
    args.opt_eps = 1e-8
    args.opt_betas = None
    args.clip_grad = None
    args.momentum = 0.9
    args.weight_decay = 0.05

    # Learning rate schedule parameters
    args.sched = 'cosine'
    args.lr = 5e-4
    args.lr_noise = None
    args.lr_noise_pct = 0.67
    args.lr_noise_std = 1.0
    args.warmup_lr = 1e-6
    args.min_lr = 1e-5
    args.decay_epochs = 30
    args.warmup_epochs = 5
    args.cooldown_epochs = 10
    args.patience_epochs = 10
    args.decay_rate = 0.1

    # Augmentation parameters
    args.color_jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.smoothing = 0.1
    args.train_interpolation = 'bicubic'
    args.repeated_aug = True

    # * Random Erase params
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1
    args.resplit = False

    # * Mixup params

    args.mixup = 0.8
    args.cutmix = 1.0
    args.cutmix_minmax = None # float
    args.mixup_prob = 1.0
    args.mixup_switch_prob = 0.5
    args.mixup_mode = 'batch'


    #----------------#
    args.dist_url = 'tcp://localhost:12613'
    args.dist_backend = 'nccl'

    args.world_size = 1


    args.print_freq = 10
    args.save_freq = 10

    args.rank = 0
    args.distributed = False


    args.exclude_file_list= ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']



    return args