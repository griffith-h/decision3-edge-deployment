import sys
sys.path[:0]=['/release','/base','/work']
import torch
torch.set_num_threads(3)
from test_math import run
run()
