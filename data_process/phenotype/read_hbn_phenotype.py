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

label_path = '/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/HBN/Phenotypic/Diagnosis_ClinicianConsensus.csv'
demo_path = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/Phenotypic/BasicDemos_Apr2024.csv'

labels = pd.read_csv(label_path)
demos = pd.read_csv(demo_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['Identifiers']

neuro, l1, l2, l3, l4, l5, l6, l7, l8 = labels['Diagnosis_ClinicianConsensus,DX_01_Cat'], labels[
    'Diagnosis_ClinicianConsensus,DX_01'], labels[
    'Diagnosis_ClinicianConsensus,DX_02'], labels['Diagnosis_ClinicianConsensus,DX_03'], labels[
    'Diagnosis_ClinicianConsensus,DX_04'], labels['Diagnosis_ClinicianConsensus,DX_05'], labels[
    'Diagnosis_ClinicianConsensus,DX_06'], labels['Diagnosis_ClinicianConsensus,DX_07'], labels[
    'Diagnosis_ClinicianConsensus,DX_08']


demo_name, age, sex, site = demos['Identifiers'], demos['Basic_Demos,Age'], demos['Basic_Demos,Sex'], demos['Basic_Demos,Study_Site']


flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []


my_dict = {
    'id': demo_name,
    'sex': sex,
    'age': age,
    'disease': [l1, l2, l3, l4, l5, l6, l7, l8]
}




# data_dict = {"subjectID": sub_name_mat, 'neuro': labels_adhd_mat}

# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/phenotype/hbn.npy', my_dict)

print('ok')
