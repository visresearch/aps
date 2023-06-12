import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss import SoftTargetCrossEntropy


def contrastive_loss(q, k, T, labels, offset, mixup , distributed):
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
        
    if distributed:
        k = concat_all_gather(k) # distribute
    logits = torch.einsum('nc,mc->nm', [q, k]).cuda()
    if mixup :
        criterion = SoftTargetCrossEntropy()        
        loss=criterion(logits / T, labels) 
    else:
        loss = F.cross_entropy(logits / T, labels) 

    similarity_pos = torch.mean(torch.diag(logits, diagonal=offset))
        
    return loss, similarity_pos


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output