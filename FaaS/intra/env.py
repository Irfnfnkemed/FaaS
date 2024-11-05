import os


def rank():
    return int(os.getenv("RANK", "0"))


def world_size():
    return int(os.getenv("WORLD_SIZE", "1"))
