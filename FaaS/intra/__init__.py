import signal
import sys

import torch.distributed as dist

from .elements import *
from .env import *
from .job import *

if not dist.is_available() or not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=env.world_size(), rank=env.rank())


def sigterm_handler(signum, frame):
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    sys.exit(0)  # 执行完毕后退出


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGKILL, sigterm_handler)
