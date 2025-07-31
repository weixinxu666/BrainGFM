import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure


import warnings
warnings.filterwarnings("ignore")



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



def get_correlation_matrices(time_series, thres):
    correlation_matrices = np.corrcoef(time_series.T) #T将数组进行转置，转成(n_rois, n_timepoints)
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

    return np.array(roi_ts).astype('float64')


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
    labels_list = []

    # labels_path = '/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/EMBARC/embarc_labels_sex.npy'
    labels_path = '/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/EMBARC/embarc_labels_hamd.npy'
    labels_mat = np.load(labels_path, allow_pickle=True).item()
    labels_subjectID = labels_mat['subjectID']
    labels_ = labels_mat['labels']
    dict_labels_id = {key: value for key, value in zip(labels_subjectID, labels_)}

    name_list = os.listdir(file_dir)

    # for site_name in tqdm(site_list[:1]):
    for name in tqdm(name_list):

        dir_path = os.path.join(file_dir, name)
        # site_dir_list = os.listdir(site_dir_path)


        sub_name = name.split('_')[0].split('-')[1][:6]

        time_series = read_singal_fmri_ts(dir_path)
        time_series = time_series[:,:100]
        time_series = time_series.T

        # cor = get_correlation_matrices(time_series, thres=0.5)
        cor = calculate_pearson_correlation(time_series)


        try:
            labels_list.append(dict_labels_id[sub_name])
            sub_name_list.append(sub_name)
            print(time_series.shape)
            cor_list.append(cor)
        except KeyError:
            print(sub_name, 'not in')


    return sub_name_list, time_series_list, cor_list, site_name_list, labels_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/Resting_fMRI_Connectivity/EMBARC_rsfMRI_fMRIPrep_Schaefer100_tp1_Baseline'

    sub_name_list, time_series_list, cor_list, site_name_list, labels_list = get_fmri_data(file_dir)



    id_mat = np.array(sub_name_list)

    conn_mat = np.array(cor_list).astype(np.float32)

    label_mat = np.array(labels_list)

    id_conn_dict = {"id": id_mat, "conn": conn_mat, 'embarc': label_mat}


    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_embarc_hamd.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_embarc_sex.npy', id_conn_dict)


    print('done')


