import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_path not in sys.path:
    sys.path.append(project_path)

from FaaS.inter.job import Job

if __name__ == '__main__':
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) + '/main.py'
    job = Job()
    job.run('127.0.0.1', 12345, script_path, [])
