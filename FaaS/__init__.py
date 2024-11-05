import torch.distributed as dist

from .intra import * 

if not dist.is_available() or not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=env.world_size(), rank=env.rank())