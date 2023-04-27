from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    pass

def get_predictions(model, data_loader):
    pass
    
