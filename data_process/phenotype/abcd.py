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

demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABCD/ABCD-3165_participants_demographic_socioeconomic_neurocogPC.tsv'

# labels = pd.read_csv(demo_path)
labels = pd.read_csv(demo_path, sep='\t')   # for .tsv files

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['participant_id']

sex, age, handedness, sib, edu, parent_edu, income = labels['sex'], labels[
    'age'], labels['handedness'], labels[
    'siblings_twins'], labels[
    'participant_education'], labels['parental_education'], labels['income']



flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []


my_dict = {
    'id': subject,
    'age': age,
    'handedness': handedness,
    'siblings_twins': sib,
    'education': edu,
    'parental_education': parent_edu,
    'income': income
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/abcd.npy', my_dict)

print('ok')
