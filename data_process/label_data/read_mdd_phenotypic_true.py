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
label_path = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/Diagnosis_ClinicianConsensus_Apr2024.csv'

labels = pd.read_csv(label_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['Identifiers']

neuro, l1, l2, l3, l4, l5, l6, l7, l8 = labels['Diagnosis_ClinicianConsensus,DX_01_Cat'], labels[
    'Diagnosis_ClinicianConsensus,DX_01'], labels[
    'Diagnosis_ClinicianConsensus,DX_02'], labels['Diagnosis_ClinicianConsensus,DX_03'], labels[
    'Diagnosis_ClinicianConsensus,DX_04'], labels['Diagnosis_ClinicianConsensus,DX_05'], labels[
    'Diagnosis_ClinicianConsensus,DX_06'], labels['Diagnosis_ClinicianConsensus,DX_07'], labels[
    'Diagnosis_ClinicianConsensus,DX_08']

flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []

hc = []

for i, sub_name in enumerate(tqdm(subject)):
    strings = [l1[i], l2[i], l3[i], l4[i], l5[i], l6[i], l7[i], l8[i]]
    # strings = [l1[i]]
    # strings = [neuro[i]]

    adhd = 0


    # 遍历字符串列表
    #pattern = re.compile(r"Major Depressive Disorder")
    # pattern = re.compile(r"Autism Spectrum Disorder")
    # pattern = re.compile(r"Neurodevelopmental Disorders")
    # pattern = re.compile(r"Generalized Anxiety Disorder")

    #pattern = re.compile(r"ADHD-Inattentive Type")
    #pattern = re.compile(r"Language Disorder")
    # pattern = re.compile(r"Intellectual")
    # pattern = re.compile(r"Posttraumatic")
    # pattern = re.compile(r"Encopresis")
    # pattern = re.compile(r"Speech Sound")

    # pattern = re.compile(r"Tourettes")

    # pattern = re.compile(r"Dysthymia")
    # pattern = re.compile(r"Provisional")
    # pattern = re.compile(r"Posttraumatic")

    # pattern = re.compile(r"Adjustment")

    #pattern = re.compile(r"Neurodevelopmental")
    # pattern = re.compile(r"Persistent (Chronic) Motor")
    # pattern = re.compile(r"Enuresis")
    # pattern = re.compile(r"Elimination")
    # pattern = re.compile(r"Trauma")
    # pattern = re.compile(r"Obsessive-Compulsive")

    #pattern = re.compile(r"Communication")
    # pattern = re.compile(r"Insomnia")
    # pattern = re.compile(r"Intermittent Explosive")
    # pattern = re.compile(r"Sleep")
    # pattern = re.compile(r"Developmental Coordination")
    pattern = re.compile(r"Feeding")

    #pattern = re.compile(r"Motor")
    #pattern = re.compile(r"Oppositional Defiant Disorder")
    #pattern = re.compile(r"Specific Learning Disorder")

    pattern2 = re.compile(r"No Diagnosis Given")

    # 遍历字符串列表
    for item in strings:
        s = str(item)
        # print(s)
        if pattern.search(s):
            adhd = 1
            all.append([sub_name.split(',')[0], adhd])
            sub.append(sub_name.split(',')[0])
            phenotype.append(adhd)
            flag_ad += 1
            break  # 如果找到"ADHD"，就不再继续查找
        # else:
        elif pattern2.search(s):
            # elif l1[i] == 'No Diagnosis Given':
            if s != 'No Diagnosis Given: Incomplete Eval':
                adhd = 0
                # print(s)
                all.append([sub_name.split(',')[0], adhd])
                sub.append(sub_name.split(',')[0])
                phenotype.append(adhd)
                flag_hc += 1
                break


# print(sub)
# print(flag_ad)
# print(flag_hc)

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
# np.save('./hbn_id_labels_2024.npy', data_dict)
# np.save('./hbn_ptsd_labels_2024.npy', data_dict)
# np.save('./hbn_odd_labels_2024.npy', data_dict)
# np.save('./hbn_adj_labels_2024.npy', data_dict)
# np.save('./hbn_neuro_labels_2024.npy', data_dict)
# np.save('./hbn_pmvt_labels_2024.npy', data_dict)
# np.save('./hbn_ptd_labels_2024.npy', data_dict)
# np.save('./hbn_psd_labels_2024.npy', data_dict)
# np.save('./hbn_eus_labels_2024.npy', data_dict)
# np.save('./hbn_ecp_labels_2024.npy', data_dict)
# np.save('./hbn_ssd_labels_2024.npy', data_dict)
# np.save('./hbn_tou_labels_2024.npy', data_dict)
# np.save('./hbn_dys_labels_2024.npy', data_dict)
# np.save('./hbn_eli_labels_2024.npy', data_dict)
# np.save('./hbn_ts_labels_2024.npy', data_dict)
# np.save('./hbn_ocd_labels_2024.npy', data_dict)
# np.save('./hbn_cd_labels_2024.npy', data_dict)
# np.save('./hbn_isd_labels_2024.npy', data_dict)  #too small sample size
# np.save('./hbn_ied_labels_2024.npy', data_dict)  #too small sample size
# np.save('./hbn_md_labels_2024.npy', data_dict)
# np.save('./hbn_dcd_labels_2024.npy', data_dict)
np.save('./hbn_fed_labels_2024.npy', data_dict)

print('ok')
