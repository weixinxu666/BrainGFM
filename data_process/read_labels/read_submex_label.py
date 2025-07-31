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

# label_path = '/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/HBN/Phenotypic/Diagnosis_ClinicianConsensus.csv'
label_path = '/home/xinxu/Lehigh/Codes/BICLab_data/SubstanceUseDisorder_Mexico_Cocaine_rsfMRI/participants.csv'

labels = pd.read_csv(label_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['participant_id']

dx = labels['group']

# neuro, l1, l2, l3, l4, l5, l6, l7, l8 = labels['Diagnosis_ClinicianConsensus,DX_01_Cat'], labels[
#     'group'], labels[
#     'Diagnosis_ClinicianConsensus,DX_02'], labels['Diagnosis_ClinicianConsensus,DX_03'], labels[
#     'Diagnosis_ClinicianConsensus,DX_04'], labels['Diagnosis_ClinicianConsensus,DX_05'], labels[
#     'Diagnosis_ClinicianConsensus,DX_06'], labels['Diagnosis_ClinicianConsensus,DX_07'], labels[
#     'Diagnosis_ClinicianConsensus,DX_08']

flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []




# 遍历字符串列表
for i in tqdm(range(len(subject))):
    if not np.isnan(dx[i]):
        sub.append(subject[i])
        phenotype.append(int(dx[i]))


phenotype = [0 if item == 1 else 1 if item == 2 else item for item in phenotype]


sub_name_mat = np.array(sub)
labels_adhd_mat = np.array(phenotype)

data_dict = {"subjectID": sub_name_mat, 'labels': labels_adhd_mat}



# np.save('./hbn_adhd_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
# np.save('./hbn_mdd_labels_2024.npy', data_dict)
# np.save('./hbn_asd_labels_2024.npy', data_dict)

# np.save('./hbn_axt_labels_2024.npy', data_dict)
# np.save('./hbn_adhd_labels_2024.npy', data_dict)
# np.save('./hbn_sld_labels_2024.npy', data_dict)
# np.save('./hbn_ld_labels_2024.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/SubstanceUseDisorder_Mexico_Cocaine_rsfMRI/submex_labels_2024.npy', data_dict)

print('ok')
