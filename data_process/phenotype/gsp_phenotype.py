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

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/BrainGenomicsSuperstructProject(GSP)_Harvard/GSP_extended_140630.csv'

labels = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['Subject_ID']

sex, age, hand, edu, race = labels['Sex'], labels[
    'Age_Bin'], labels[
    'Hand'], labels['Educ'], labels[
    'Race_Ethn']



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
    'handedness': hand,
    'edu': edu,
    'race': race
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/gsp.npy', my_dict)

print('ok')
