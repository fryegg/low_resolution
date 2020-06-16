import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch

# Seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
epoch = 100
# Basic Python Environment
python = '/home/gpu/anaconda3/envs/fryegg/bin/python'
gpu = '1,2,3,4,5'
# Hyperparameter Candidate
layer = [1, 2, 3, 4, 5]
sizes = [32,16,8]
gamma = 1.5
#gamma = np.random.choice([0.5], size=num_size, replace=True)

#Plain
def tr_plain_gpu(size):
    # with no distillation
    script1 = '%s main.py --train_type plain --size %d --epoch %d --gpu %s' % (python,size,epoch,gpu)
    subprocess.call(script1, shell=True)

# Distill
def tr_distill_gpu(ix, size,gamma):
    # with distillation
    script2 = '%s main.py --train_type distill --size %d --layer %d --epoch %d --gpu %s --gamma %f' % (python, size,layer[ix-1],epoch,gpu,gamma)
    subprocess.call(script2, shell=True)

for size in sizes:
    tr_plain_gpu(size)
    for ix in layer:
        tr_distill_gpu(ix, size,gamma)
