import signal
import sys

import torch.distributed as dist

from .elements import *
from .env import *
from .job import *

def sigterm_handler(signum, frame):
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    sys.exit(0)  # 执行完毕后退出


signal.signal(signal.SIGTERM, sigterm_handler)
