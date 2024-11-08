import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_path not in sys.path:
    sys.path.append(project_path)
    
from FaaS.inter.alloc import Alloc

if __name__ == '__main__':
    scheduler = Alloc([0,1,2,3])
    scheduler.run()