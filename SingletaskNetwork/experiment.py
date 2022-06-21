import pandas as pd
import numpy as np
from pathlib import Path
import os

data_dir = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned')
training_df_folder = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/Lists/LSO_Training_Lists/')
df = pd.read_csv('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/FID_Data_SingleTask_Prep_Only2Class.csv')

# df = df[df.columns[:30]]


files_exist = [path.stem.split('_T1')[0] for path in list(data_dir.glob('*.gz'))]
df = df[df.MRID.isin(files_exist)]


categorical_cols = [
    'Diagnosis Interpolated',
    'APOE Genotype Interpolated',
    'Diabetes Interpolated',
    'Hyperlipidemia Interpolated',
    'Hypertension Interpolated',
    'Diagnosis Depression Interpolated',
    'Smoking Interpolated'
]


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

def get_min_sampling(df, cat_col, limit_samples = None):
    df_tmp = df.copy()
    df_tmp = df_tmp.dropna(subset = cat_col)

    df_tmp[cat_col] = df[cat_col].astype('category').cat.codes
    min_samples = df_tmp[cat_col].value_counts().min()
    if limit_samples:
        if min_samples > limit_samples: 
            min_samples=limit_samples

    df_tmp = pd.concat([
        df_tmp[df_tmp[cat_col]==val].sample(min_samples, random_state=42)
        for val in df_tmp[cat_col].unique()
    ]).reset_index(drop=True)


    df_tmp = df_tmp[['MRID', cat_col, 'Study']]
    df_tmp.columns = ['MRID', 'Label', 'Study']
    return df_tmp

def get_regression_df(df, reg_col, limit_samples = None):
    df_tmp = df.copy()
    df_tmp = df_tmp.dropna(subset = reg_col).reset_index(drop=True)
    if limit_samples:
        if df_tmp[reg_col].count() > limit_samples: 
            df_tmp=df_tmp.sample(limit_samples, random_state=42)

    df_tmp[reg_col] = df_tmp[reg_col].astype('float32')
    df_tmp = df_tmp[['MRID', reg_col, 'Study']]
    df_tmp.columns = ['MRID', 'Label', 'Study']
    return df_tmp





## Leave-Task-Out Training Setup 
## LSO training files for categorical target

training_df_folder.mkdir(exist_ok=True)
for sel_col in categorical_cols:
    df_task_sel = get_min_sampling(df, sel_col, limit_samples=4000)
    df_filt = df[~df.MRID.isin(df_task_sel.MRID)]

    curr_folder = training_df_folder / sel_col.replace(" ","_")
    curr_folder.mkdir(exist_ok=True)
    df_task_sel.to_csv(curr_folder / f'{sel_col.replace(" ","_")}_MAIN.csv', index=False)

    for cat_col_test in list(set(categorical_cols)-set([sel_col])):

        df_tmp = get_min_sampling(df_filt, cat_col_test)
        df_tmp.to_csv(curr_folder / f'{cat_col_test.replace(" ","_")}_CHILD.csv', index=False)
        
    for reg_col_test in list(set(regression_cols)-set([sel_col])):
        
        df_tmp = get_regression_df(df_filt, reg_col_test)
        df_tmp.to_csv(curr_folder / f'{reg_col_test.replace(" ","_")}_CHILD_REG.csv', index=False)

## LSO training files for regression target
for sel_col in regression_cols:
    df_task_sel = get_regression_df(df, sel_col, limit_samples=8000)
    df_filt = df[~df.MRID.isin(df_task_sel.MRID)]

    curr_folder = training_df_folder / (sel_col.replace(" ","_")+'_REG')
    curr_folder.mkdir(exist_ok=True)
    df_task_sel.to_csv(curr_folder / f'{sel_col.replace(" ","_")}_MAIN_REG.csv', index=False)

    for cat_col_test in list(set(categorical_cols)-set([sel_col])):

        df_tmp = get_min_sampling(df_filt, cat_col_test)
        df_tmp.to_csv(curr_folder / f'{cat_col_test.replace(" ","_")}_CHILD.csv', index=False)
        
    for reg_col_test in list(set(regression_cols)-set([sel_col])):
        
        df_tmp = get_regression_df(df_filt, reg_col_test)
        df_tmp.to_csv(curr_folder / f'{reg_col_test.replace(" ","_")}_CHILD_REG.csv', index=False)


# model_size = 18
# batch_size = 8
# experiment_tag = 'Test_LSO'
# leave_site_out = True
# main_task = ''
# lso_task_type = ''




# task_sel = ['Age_REG', 'Hypertension_Interpolated']

# for task in task_sel:
#     task_folder = training_df_folder / task
#     main_task = task

#     for csv_path in list(task_folder.glob('*.csv')):
#         name = csv_path.stem
#         print(name + '_ResNet' + str(model_size))
#         if 'REG' in name:
#             type = 'Regression'
#         else:
#             type = 'Classification'
        
#         if 'CHILD' in name:
#             lso_task_type = 'Subtask'
#         elif 'MAIN' in name:
#             lso_task_type = 'Baseline'

#         df_tmp = pd.read_csv(csv_path)
#         samples = df_tmp.shape[0]
#         epochs = 7 + int(10000/samples)

#         submission_string = f"""
#             qsub -l gpu=1 -l h_vmem=64G -pe threaded 8 submit_lso.sh \
#             {name} \
#             {csv_path} \
#             {model_size} \
#             {batch_size} \
#             {name.split("_Inter")[0]} \
#             {type} \
#             {experiment_tag} \
#             {leave_site_out} \
#             {main_task} \
#             {lso_task_type} \
#             {epochs}
#         """

#         os.system(submission_string)





## Single Task Training Setup 

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


# model_sizes = [18]
# batch_sizes = [8]
# experiment_tag = 'ONLY_VAL_Augmentations'


# for model_size, batch_size in zip(model_sizes, batch_sizes):
#     for csv_path in list(training_df_folder.glob('*.csv')):
#         name = csv_path.stem
#         print(name + '_ResNet' + str(model_size))
#         if 'REG' in name:
#             type = 'Regression'
#         else:
#             type = 'Classification'

#         submission_string = f"""
#         qsub -l gpu=1 -l h_vmem=64G -pe threaded 8 submit.sh 
#         {name}
#         {csv_path} 
#         {model_size} 
#         {batch_size} 
#         {name.split("_Inter")[0]} 
#         {type} 
#         {experiment_tag}
#         """
#         os.system(submission_string)

# quit = []

# for x in quit:
#     os.system(f'qdel {str(x)}')
