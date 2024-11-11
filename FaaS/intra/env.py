import os


def rank():
    return int(os.getenv("RANK", "0"))

def local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))

def world_size():
    return int(os.getenv("WORLD_SIZE", "1"))
