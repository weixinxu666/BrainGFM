import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure


# import warnings
# warnings.filterwarnings("ignore")


def calculate_pearson_correlation(matrix):
    """
    计算输入矩阵的脑区之间的 Pearson 相关系数矩阵。

    参数:
    matrix (np.ndarray): 形状为 (n_regions, n_timepoints) 的矩阵，
                         其中 n_regions 是脑区数量，n_timepoints 是时间序列长度。

    返回:
    np.ndarray: 形状为 (n_regions, n_regions) 的相关系数矩阵。
    """
    # 创建 ConnectivityMeasure 对象，指定 kind='correlation' 使用 Pearson 相关系数
    correlation_measure = ConnectivityMeasure(kind='correlation')

    # 输入矩阵进行计算，结果为一个相关系数矩阵
    correlation_matrix = correlation_measure.fit_transform([matrix.T])[0]

    return correlation_matrix


def get_correlation_matrices(time_series):
    correlation_matrices = np.corrcoef(time_series) #T将数组进行转置，转成(n_rois, n_timepoints)
    # threshold = thres  # 阈值，用于过滤弱连接
    # adj_matrix = np.zeros_like(correlation_matrices)
    # adj_matrix[correlation_matrices >= threshold] = 1
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = np.expand_dims(adj_matrix, axis=0)
    return correlation_matrices  # 100x100



def read_singal_fmri_ts(data_file):
    csv_data = pd.read_csv(data_file, header=None)  # 读取训练数据
    data = np.array(csv_data)
    roi_ts = []
    for d in data:
        ts = d[0].split(' ')
        roi_ts.append(ts)

    return np.array(roi_ts).astype('float32')


def unify_time_roi_matrices(time_series_roi, thres):
    time_size = time_series_roi.shape[0]
    if time_size <= thres:
        new_matrix = np.zeros((thres, 100))
        new_matrix[:time_size, :] = time_series_roi
        return new_matrix
    else:
        new_matrix = time_series_roi[:thres, :]
        return new_matrix

def get_fmri_data(file_dir):
    sub_name_list = []
    time_series_list = []
    cor_list = []
    site_name_list = []

    id_list = []
    conn_list = []
    nan_list = []
    label_list = []

    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_mdd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_asd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_axt_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_adhd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_id_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_ssd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_ld_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_ptsd_labels_2024.npy'
    # embarc_sex_labels = './hbn_cd_labels_2024.npy'
    # embarc_sex_labels = './hbn_isd_labels_2024.npy'
    embarc_sex_labels = './hbn_dcd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_neuro_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_eli_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_ocd_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_sld_labels_2024.npy'
    # embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_odd_labels_2024.npy'

    labels_sex = np.load(embarc_sex_labels, allow_pickle=True).item()

    labels_id_list = labels_sex['subjectID']
    labels_sex_list = labels_sex['labels']

    dict_labels_id = {key: value for key, value in zip(labels_id_list, labels_sex_list)}

    site_list = os.listdir(file_dir)

    # for site_name in tqdm(site_list[:1]):
    for site_name in tqdm(site_list):

        site_dir_path = os.path.join(file_dir, site_name, 'ROISignals_AROMA')
        site_dir_list = os.listdir(site_dir_path)
        for i in tqdm(site_dir_list):
            sub_dir_path = os.path.join(site_dir_path, i)
            subject_dir_list = os.listdir(sub_dir_path)
            for csv_name in subject_dir_list:
                id_indiv = csv_name.split('_')[0].split('-')[-1]
                csv_path = os.path.join(sub_dir_path, csv_name)
                time_series = read_singal_fmri_ts(csv_path)
                time_series = time_series[:,:100]
                time_series = time_series.T
                # time_series_100 = get_correlation_matrices(time_series, thres=0)
                conn = get_correlation_matrices(time_series)
                # conn = calculate_pearson_correlation(time_series)

                # sub_name_list.append(sub_name)
                # print(time_series.shape)
                # time_series_list.append(time_series)
                # # print(time_series_100.shape)
                # cor_list.append(cor)
                # site_name_list.append(site_name)

                if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                    nan_list.append(id_indiv)
                else:
                    try:
                        label_list.append(dict_labels_id[id_indiv])
                        id_list.append(id_indiv)
                        conn_list.append(conn)
                    except KeyError:
                        print(id_indiv, 'not in')

    return id_list, time_series_list, conn_list, site_name_list, label_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/rsfMRI/rsfMRI_Preprocessed_ROISignals_Schaefer135ROIs_AlexZhao'

    sub_name_list, time_series_list, cor_list, site_name_list, label_list = get_fmri_data(file_dir)

    id_mat = np.array(sub_name_list)
    # time_series_mat = np.array(time_series_list)
    conn_mat = np.array(cor_list).astype(np.float32)
    label_mat = np.array(label_list)


    id_conn_dict = {"id": id_mat, "conn": conn_mat, 'label': label_mat}


    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_mdd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_axt.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_adhd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_id.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_ssd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_ld.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_ptsd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_cd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_isd.npy', id_conn_dict)
    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_dcd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_neuro.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_eli.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_ocd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_odd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_hbn_sld.npy', id_conn_dict)


    print('done')


