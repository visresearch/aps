
import os
import shutil
import random
import numpy as np
import torch
import torch.distributed as dist


def fix_random_seeds(seed=31):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        # self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


    def synchronize_between_processes(self):  
        # pack = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        pack = torch.tensor([self.sum, self.count], device='cuda')
        dist.barrier()
        dist.all_reduce(pack)
        self.sum, self.count = pack.tolist()

    
    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):

        fmtstr = '{} {' + self.fmt + '} ({' + self.fmt + '})'
        return fmtstr.format(self.name, self.val, self.avg)



if __name__ == '__main__':
    import numpy as np
    meter = AverageMeter('test', ':.4e')

    a = np.arange(10.0)

    for i in range(10):
        meter.update(a[i], 1)
        print(meter)
    
