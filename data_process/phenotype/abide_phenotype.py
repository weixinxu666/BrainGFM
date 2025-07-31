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

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABIDE/ABIDE_II.csv'

labels = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['SUB_ID']

sex, age, hand, dx, site, fiq, viq, piq = labels['SEX'], labels[
    'AGE_AT_SCAN '], labels[
    'HANDEDNESS_CATEGORY'], labels['DX_GROUP'], labels['SITE_ID'], labels[
    'FIQ'], labels['VIQ'], labels[
    'PIQ']



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
    'site': site,
    'fiq': fiq,
    'viq': viq,
    'piq': piq
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/abide2.npy', my_dict)

print('ok')
