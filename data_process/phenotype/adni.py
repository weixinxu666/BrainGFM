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

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/ADNIMERGE.csv'

labels = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['RID']

sex, age, disease, edu, race, ethic, marry, abeta, tau, ptau = labels['PTGENDER'], labels[
    'AGE'], labels['DX_bl'], labels['PTEDUCAT'], labels[
    'PTRACCAT'], labels['PTETHCAT'], labels['PTMARRY'], labels['ABETA'], labels['TAU'], labels['PTAU']



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
    'disease': disease,
    'edu': edu,
    'ethic': ethic,
    'race': race,
    'marry': marry,
    'abeta': abeta,
    'tau': tau,
    'ptau': ptau,
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/adni.npy', my_dict)

print('ok')
