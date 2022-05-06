from sklearn.model_selection import StratifiedKFold,  train_test_split
import torch
import pandas
import numpy as np
from data import Dataset_3d
from model import Model_3d, resnet50, resnet34
from train import Trainer

# Setup for GPU Acceleration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

print("Torch GPUs: " + str(torch.cuda.device_count()))
print(torch.get_num_threads())
params = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4}

params_test = {
    'batch_size': 4,
    'shuffle': False,
    'num_workers': 1}

max_epochs = 20
learning_rate = 0.00035
momentum = 0.99


net_path = "./weights/"
data_dir = '/cbica/home/bashyamv/comp_space/3_DataSets/LifeSpanCN/Data/'
df = pandas.read_csv('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/train_age_df.csv')

def get_per_target_heads(training_set, targets)
    ## Get the number of output nodes needed for each task
    per_target_heads = []
    for target in targets:
        if list(target.values())[0]['Type'] == 'Categorical':
            per_target_heads.append(len(training_set.labels[(list(target.keys())[0])].dropna().unique()))
        else:
            per_target_heads.append(1)
        return per_target_heads


skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(df, df.SEX)

fold=0
for train_index, test_index in skf.split(df, df.SEX):

    fold += 1
    train_index, val_index = train_test_split(train_index, test_size=0.2, )
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[val_index].reset_index(drop=True)
    test_cv_df = df.iloc[test_index].reset_index(drop=True)

    training_set = Dataset_3d(train_df, data_dir)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_set = Dataset_3d(val_df, data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_test)
    testing_cv_set = Dataset_3d(test_cv_df, data_dir)
    testing_cv_generator = torch.utils.data.DataLoader(testing_cv_set, **params_test)

    per_target_heads = get_per_target_heads(training_set, training_set.targets)
    # network = torch.nn.DataParallel(Model_3d(8, 5)).to(device)
    network = resnet34( sample_size= 182, sample_duration = 182).to(device)
    # network = Model_3d(8, 2).to(device)

    # network.load_state_dict(torch.load("./weights/20_model.pkl"))
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

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
        device = device)

    trainer.train_loop(
        validation = True,
        save_models = True,
        amp = False)

    trainer.test_loop(
        testing_cv_generator,
        testing_cv_set,
        './Pred/Test_Balanced_'+str(fold)+'.csv')