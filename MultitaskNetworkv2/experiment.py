import pandas as pd
import numpy as np
from pathlib import Path
import os

data_dir = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned')
training_df_folder = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/Lists/Training_Lists')
df_path = '/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/FID_Data_SingleTask_Prep_Only2Class.csv'
df = pd.read_csv(df_path)

# df = df[df.columns[:30]]


files_exist = [path.stem.split('_T1')[0] for path in list(data_dir.glob('*.gz'))]
df = df[df.MRID.isin(files_exist)]


# categorical_cols = [
#     'Diagnosis Interpolated',
#     'APOE Genotype Interpolated',
#     'Diabetes Interpolated',
#     'Hyperlipidemia Interpolated',
#     'Hypertension Interpolated',
#     'Diagnosis Depression Interpolated',
#     'Smoking Interpolated'
# ]


regression_cols = [
    'Age',
    'MMSE Interpolated',
    'Education Years Interpolated',
    'CDR Global Interpolated', 
    'Abeta CSF Interpolated',
    'Tau CSF Interpolated',
    'BMI Interpolated',
    'Digit Span Forward Interpolated',
    'Fluid Intelligence Interpolated',
    'BNT Interpolated'
    ]

# def get_min_sampling(df, cat_col):
#     df_tmp = df.copy()
#     df_tmp = df_tmp.dropna(subset = cat_col)

#     df_tmp[cat_col] = df[cat_col].astype('category').cat.codes
#     min_samples = df_tmp[cat_col].value_counts().min()

#     df_tmp = pd.concat([
#         df_tmp[df_tmp[cat_col]==val].sample(min_samples, random_state=42)
#         for val in df_tmp[cat_col].unique()
#     ]).reset_index(drop=True)
#     df_tmp = df_tmp[['MRID', cat_col, 'Study']]
#     df_tmp.columns = ['MRID', 'Label', 'Study']
#     return df_tmp

def get_regression_df(df, reg_col):
    df_tmp = df.copy()
    df_tmp = df_tmp.dropna(subset = reg_col).reset_index(drop=True)
    df_tmp[reg_col] = df_tmp[reg_col].astype('float32')
    df_tmp = df_tmp[['MRID', reg_col, 'Study']]
    df_tmp.columns = ['MRID', 'Label', 'Study']
    return df_tmp



# # Write Categorical Dataframes to csvs
# training_df_folder.mkdir(exist_ok=True)
# for cat_col in categorical_cols:
#     df_tmp = get_min_sampling(df, cat_col)
#     df_tmp.to_csv(training_df_folder / f'{cat_col.replace(" ","_")}.csv', index=False)

# # Write Regression Dataframes to csvs
# training_df_folder.mkdir(exist_ok=True)
# for reg_col in regression_cols:
#     print(reg_col)
#     df_tmp = get_regression_df(df, reg_col)
#     df_tmp.to_csv(training_df_folder / f'{reg_col.replace(" ","_")}_REG.csv', index=False)


model_sizes = [18]
batch_sizes = [8]



for model_size, batch_size in zip(model_sizes, batch_sizes):
    name = 'V2Model_Multitask_10batchAgg'
    print(name + '_ResNet' + str(model_size))
    os.system(f'qsub -l gpu=1 -l h_vmem=64G -pe threaded 8 submit.sh {name} {df_path} {model_size} {batch_size}')


# quit = []

# for x in quit:
#     os.system(f'qdel {str(x)}')
