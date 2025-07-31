import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm
import re

import warnings

warnings.filterwarnings("ignore")

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/ADHD200/ADHD200_Phenotypic.csv'

labels = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['Participant ID']

sex, age, hand, dx, index, inatt, hpimp, iq = labels['Gender'], labels[
    'Age'], labels[
    'Handedness'], labels['DX'], labels[
    'ADHD Index'], labels['Inattentive'], labels[
    'Hyper/Impulsive'], labels['Full4 IQ']



flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []


my_dict = {
    'id': subject,
    'sex': sex,
    'age': age,
    'disease': dx,
    'handedness': hand,
    'adhd index': index,
    'Inattentive': inatt,
    'Hyper/Impulsive': hpimp,
    'IQ': iq
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/adhd200.npy', my_dict)

print('ok')
