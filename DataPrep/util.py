import pandas as pd
import numpy as np
import os.path
from tqdm import tqdm
from multiprocessing import Pool
import time


def checkFileExist(df, col_names, threads):

    #############################################
    #Function to check if files in a list exist
    #Parameters:
    #  df = Datafrom of file paths
    #  col_names = list of columns to check
    #Returns:
    #   None
    #   Prints number of files found and missing
    #############################################
    count_found = 0
    count_missing = 0

    pbar = tqdm()
    pbar.reset(total=len(col_names))

    for col in col_names:
        print("\n\nChecking " + col)
        p = Pool(threads)
        found = p.map(checkFileExistSingle,df[col], chunksize = 1000)

        p.close()
        p.join()
        pbar.update()
        print('---'+ col + '---\nNumber of files found: ' + str(sum(found)) + ' \nPercentage of files found ' + str(round(100*sum(found)/len(df), 4)))



def checkFileExistSingle(file_paths):

    #############################################
    #Function to check if files in a list exist
    #Parameters:
    #  file_paths = list or pandas series
    #Returns:
    #   count_found = Number of files found
    #############################################
    count_found = 0
    count_missing = 0



    try:
        if os.path.isfile(file_paths):
            count_found += 1
        else:
            print('File Not Found: ' + file_paths)
            count_missing += 1
    except Exception as e:
        print(e)
        print(file_paths)
    return count_found
