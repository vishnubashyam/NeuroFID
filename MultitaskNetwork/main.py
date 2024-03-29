from sklearn.model_selection import StratifiedKFold,  train_test_split
import torch
import pandas
import numpy as np
from data import Dataset_ROI
from model_ROI import Model_ROI
from train import Trainer
import json

# Setup for GPU Acceleration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


f = open('./training_config.json')
targets = json.load(f)['Target_Varibles']

print("Torch GPUs: " + str(torch.cuda.device_count()))
print(torch.get_num_threads())
params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 4}

params_test = {
    'batch_size': 256,
    'shuffle': False,
    'num_workers': 1}

max_epochs = 20
learning_rate = 3e-2
momentum = 0.99


net_path = "./weights/"
df = pandas.read_csv('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/Lists/FID_v2_DataList_with_ROIs.csv')


skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(df, df.Study)

fold=0
for train_index, test_index in skf.split(df, df.Study):

    fold += 1
    train_index, val_index = train_test_split(train_index, test_size=0.2, )
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[val_index].reset_index(drop=True)
    test_cv_df = df.iloc[test_index].reset_index(drop=True)

    training_set = Dataset_ROI(train_df, targets)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_set = Dataset_ROI(val_df, targets)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_test)
    testing_cv_set = Dataset_ROI(test_cv_df, targets)
    testing_cv_generator = torch.utils.data.DataLoader(testing_cv_set, **params_test)

    per_target_heads, target_type = training_set.per_target_heads, training_set.target_types
    network = Model_ROI(targets, per_target_heads, df.columns.str.contains('MUSE').sum()).to(device)

    # network.load_state_dict(torch.load("./weights/20_model.pkl"))
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = {
        "Numerical": torch.nn.MSELoss(),
        "Categorical": torch.nn.CrossEntropyLoss()}


    print('--Starting Training--')
    trainer = Trainer(
        max_epochs = max_epochs,
        fold = fold,
        training_generator = training_generator, 
        validation_generator = validation_generator,
        network = network,
        optimizer = optimizer,
        criterion = criterion,
        output_path = net_path,
        device = device,
        target_types = target_type,
        targets = targets
        )

    trainer.train_loop(
        validation = True,
        save_models = True,
        amp = False)

    trainer.test_loop(
        testing_cv_generator,
        testing_cv_set,
        './Pred/Test_Balanced_'+str(fold)+'.csv')