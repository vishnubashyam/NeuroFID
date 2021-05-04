import pandas as pd
import numpy as np
import os.path
from tqdm import tqdm
from multiprocessing import Pool
import time
from util import checkFileExist


columns_of_interest = ['MRID', 'PTID', 'Study', 'Age', 'Diagnosis', 'MMSE', 'Sex', 'T1', 'MUSE']
check_cols = ['T1', 'MUSE', 'BrainMask']
output_file = './ISTAGING_masterlist_FID.csv'

df = pd.read_pickle("/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/20210415/istaging.pkl.gz")
mri_df = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/20210415/MRI.pkl.gz')


df_merge = df.merge(mri_df)
df_merge = df_merge[df_merge['Study']!='WRAP'].reset_index(drop=True)
df_merge = df_merge[columns_of_interest]

df_merge = df_merge[df_merge['T1'].notna()]
df_merge = df_merge[df_merge['MUSE'].notna()]


df_merge['BrainMask'] = df_merge['MUSE'].str.replace('MUSE', 'Skull-Stripped')
df_merge['BrainMask'] = df_merge['BrainMask'].str.replace('_brain_muse-ss_Mean_fastbc_muse.nii.gz', '_brainmask_muse-ss.nii.gz')


# Greifswalf has different file names
df_merge['BrainMask'] = df_merge['BrainMask'].str.replace('brain_QCed_muse_dramms\+ants_C1.2.nii.gz', 'brainMask_QCed.nii.gz')


# Multiprocessing check for file existance
checkFileExist(df_merge, check_cols, 30)

print('Writing csv file: ' + output_file)
df_merge.to_csv(output_file, index = False)



