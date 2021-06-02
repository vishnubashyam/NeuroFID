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

print("Torch GPUs: " + str(torch.cuda.device_count()))

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 10}


max_epochs = 100
learning_rate = 0.0015
momentum = 0.99


net_path = "./weights/"
data_dir = '/cbica/home/bashyamv/comp_space/3_DataSets/LifeSpanCN/Data/'
df = pandas.read_csv('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/train_age_df.csv')


skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(df, df.SEX)

fold=0
for train_index, test_index in skf.split(df, df.SEX):

    fold += 1
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[test_index].reset_index(drop=True)

    training_set = Dataset_3d(train_df, data_dir)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset_3d(val_df, data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # network = torch.nn.DataParallel(Model_3d(8, 5)).to(device)
    network = Model_3d(8, 5).to(device)

    # network.load_state_dict(torch.load("./weights/20_model.pkl"))
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print('--Starting Training--')
    trainer = Trainer(1, fold, training_generator, validation_generator, network, optimizer, criterion)
    trainer.train_loop(amp = True)
