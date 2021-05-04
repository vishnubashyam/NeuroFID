from sklearn.model_selection import StratifiedKFold
import torch
import pandas
import numpy as np
from data import Dataset_3d
from model import Model_3d
from train import Trainer

# Setup for GPU Acceleration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
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

    fold += 1
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[test_index].reset_index(drop=True)

    training_set = Dataset_3d(train_df, data_dir)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset_3d(val_df, data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    network = nn.DataParallel(Model_3d(8, 5)).to(device)

    CUDA_LAUNCH_BLOCKING=1
    # network.load_state_dict(torch.load("./weights/20_model.pkl"))
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, momentum=momentum)
    criterion = torch.nn.BCELoss()

    trainer = Trainer(30, fold, training_generator, validation_generator, network, optimizer, criterion)
    trainer.train_loop()