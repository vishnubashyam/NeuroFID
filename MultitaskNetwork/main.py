import torch
import torchvision
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
from torchvision import datasets, models, transforms
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import Data, Train, Model


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 20}

max_epochs = 100
learning_rate = 0.0000015
momentum = 0.99


net_path = "./weights/"

data_dir = '/home/bashyamv/Research/Data/Amyloid_Scans/BrainAligned/'

df = pandas.read_csv('/home/bashyamv/Research/Data/Amyloid_Scans/train_df_3d.csv')


skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(df, df.amyloidStatus)

fold=0
for train_index, test_index in skf.split(df, df.amyloidStatus):
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[test_index].reset_index(drop=True)

    training_set = Data.Dataset_3d(train_df, data_dir)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Data.Dataset_3d(val_df, data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    network = nn.DataParallel(Model.model_3d(8, 5)).to(device)

    # network =model_3d(8, 0).to(device)/


    CUDA_LAUNCH_BLOCKING=1

    # network.load_state_dict(torch.load("./weights/20_model.pkl"))

    print(network)




    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    criterion = torch.nn.BCELoss()
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    Train.train(30, fold, training_generator, validation_generator, network, optimizer, criterion)
    fold+=1
