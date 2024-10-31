import os

def rank():
    return int(os.getenv("rank", "0"))