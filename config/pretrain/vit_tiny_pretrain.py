import argparse


def vit_tiny_pretrain():
    args = argparse.Namespace()
    args.arch = 'vit-tiny'    
    args.sampling_ratio = 0.25
    args.resume = False
    args.data_root = '/path/to/dataset/'
    args.dataset = 'cifar10'
    args.img_dim = 32
    args.nepoch = 3200

    args.warmup_epoch = 20

    args.batch_size = 512
    args.num_workers = 12
    args.nsampling = 4
    args.power = 3
    args.lr_base = 1e-3
    args.weight_decay = 0.05
    args.momentum_rate = 0.99

    args.temp = 0.1
    args.loss = 'contrastive_loss'

    args.print_freq = 10
    args.eval_freq = 5
    args.save_freq = 10

    args.prediction = True

    args.momentum = False

    args.out_dim = 128
    args.hidden_dim = 512

    args.exp_dir = f'./log/pretrain/{args.dataset}/ckpts_{args.dataset}_{args.arch}_loss_{args.loss}'\
        f'_temp{args.temp}_lr_base{args.lr_base}_batch{args.batch_size}_epoch{args.nepoch}'\
        f'_warmup{args.warmup_epoch}_sampling_ratio{args.sampling_ratio}_nsampling{args.nsampling}'\
        f'_power{args.power}/'

    args.rank = 0
    args.distributed = False
    args.use_mix_precision = True
    args.init_method = 'tcp://localhost:18998'
    args.world_size = 1

    args.mixup = 0.8
    args.cutmix = 1.0
    args.cutmix_minmax = None # float
    args.mixup_prob = 1.0
    args.mixup_switch_prob = 0.5
    args.mixup_mode = 'batch'
    args.smoothing = 0.5

    args.exclude_file_list= ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args