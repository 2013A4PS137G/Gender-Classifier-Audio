import os
import pandas as pd
import librosa
import warnings
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.io import savemat
import progressbar
from shutil import copyfile

warnings.filterwarnings("ignore")


def createDir(folder):
    if (not os.path.exists(folder)):
        os.makedirs(folder)


def copyFiles(folder_in, file_list, output_folder, samples_per_class):
    createDir(output_folder)
    print(output_folder + ' :-')
    bar = progressbar.ProgressBar(maxval=samples_per_class,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    num_samples = 0
    for index, row in file_list.iterrows():
        fname = row['filename'].split('/')[1]
        copyfile(folder_in + '/' +
                 row['filename'], output_folder + '/' + fname)
        num_samples = num_samples + 1
        bar.update(num_samples)
        if(num_samples == samples_per_class):
            break


folder_in = "Common Voice Corpus - Mozilla/cv-valid-train"
output_folder = "data"
createDir(output_folder)

df = pd.read_csv(folder_in + ".csv")
list_male = df[pd.notnull(df['age']) & (df['gender'] == 'male') & (
    df['down_votes'] == 0) & (df['up_votes'] > 2)]
list_female = df[pd.notnull(df['age']) & (df['gender'] == 'female') & (
    df['down_votes'] == 0) & (df['up_votes'] > 2)]

limit = min(len(list_male), len(list_female))

copyFiles(folder_in, list_male, output_folder + "/male", limit)
copyFiles(folder_in, list_female, output_folder + "/female", limit)
