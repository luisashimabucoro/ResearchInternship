import torch
from torch import nn
import math

from core.utils import accuracy, get_shuffle, check_gpu_memory_usage, create_split_list

for fold in range(2, 26):
    print(create_split_list(25, fold))