import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_path not in sys.path:
    sys.path.append(project_path)

from FaaS.inter.job import Job
from main import main 

import sys

if __name__ == '__main__':
    job = Job()
    job.run(main, ())


