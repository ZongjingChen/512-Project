from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer
from model import TopClusModel
import os
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
from utils import TopClusUtils
import numpy as np
from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


x = torch.randn((2, 2, 2))
# print(x)
x_ori = Parameter(x)
x = x_ori.view(-1, 2)
# print(x)
t = torch.tensor([[1.0,1.0],[1.0,1.0], [1.0,1.0],[1.0,1.0]])
print(x_ori)
print(t)
optimizer = Adam([x_ori], lr=0.1)
loss = F.mse_loss(x, t)
loss.backward()
optimizer.step()
print(x_ori)



