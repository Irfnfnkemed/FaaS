import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_path not in sys.path:
    sys.path.append(project_path)

from FaaS.inter.scheduler import Scheduler

if __name__ == '__main__':
    scheduler = Scheduler(12345)
    scheduler.set_device('127.0.0.1', [0, 1, 2, 3])
    scheduler.run()
