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

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/OASIS-3/PhenotypicData/OASIS-3_ADRC Clinical Data.csv'

labels = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['ADRC_ADRCCLINICALDATA ID']

age, disease, height, weight = labels[
    'ageAtEntry'], labels['dx1'], labels[
    'height'], labels[
    'weight']



flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []


my_dict = {
    'id': subject,
    'age': age,
    'disease': disease,
    'height': height,
    'weight': weight
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/oasis.npy', my_dict)

print('ok')
