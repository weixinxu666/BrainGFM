import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
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
    correlation_matrices = np.corrcoef(time_series) #T将数组进行转置，转成(n_rois, n_timepoints)
    # threshold = thres  # 阈值，用于过滤弱连接
    # adj_matrix = np.zeros_like(correlation_matrices)
    # adj_matrix[correlation_matrices >= threshold] = 1
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = np.expand_dims(adj_matrix, axis=0)
    return correlation_matrices  # 100x100


def cal_pearson_conn(time_series):
    # input is :  regions x time point
    n_regions = time_series.shape[0]
    connectivity_matrix = np.zeros((n_regions, n_regions))

    # 计算Pearson相关系数
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                connectivity_matrix[i, j] = 1  # 自相关为1
            else:
                # 计算Pearson相关系数
                corr, _ = pearsonr(time_series[i], time_series[j])
                connectivity_matrix[i, j] = corr
                connectivity_matrix[j, i] = corr  # 矩阵对称

    # connectivity_matrix 是最终的功能连接矩阵
    # print(connectivity_matrix)
    return connectivity_matrix



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

    id_list = []
    conn_list = []
    nan_list = []

    mat_data = scio.loadmat('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/adni23_label.mat')

    data = mat_data['data']


    roi_data = data['ROISignals_Schaefer100']
    sex = data['gender']
    sub_id = data['subjectID']
    dx = data['DX_bl']

    print(dx)

    id_list = []
    dx_list = []
    label_list = []

    sub_name_list = []

    for i in tqdm(range(len(sub_id))):
        try:
            print(dx[0])
            dx_i = dx[i][0][0]
            id_i = sub_id[i][0][0]
            id_list.append(id_i.split('_')[-1])
            dx_list.append(dx_i)
        except IndexError:
            print(id_i, 'not in')

    dx_list = [0 if label in ['CN', 'SMC'] else 1 for label in dx_list]
    # dx_list = [1 if label == 'AD' else 0 if label in ['CN', 'SMC'] else None for label in dx_list]
    # dx_list = [1 if label == 'LMCI' else 0 if label in ['CN', 'SMC'] else None for label in dx_list]

    # dx_list = [1 if label == 'AD' else 0 for label in dx_list if label != 'LMCI']
    # dx_list = [1 if label == 'LMCI' else 0 for label in dx_list if label != 'AD']

    name_list = os.listdir(file_dir)

    dict_labels_id = {key: value for key, value in zip(id_list, dx_list)}

    # for site_name in tqdm(site_list[:1]):
    for name in tqdm(name_list):

        dir_path = os.path.join(file_dir, name)
        # site_dir_list = os.listdir(site_dir_path)

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # if file.endswith("Schaefer100.csv"):
                if file.endswith("AAL116.csv"):
                    id_indiv = name.split('_')[0]
                    csv_path = os.path.join(root, file)
                    time_series = read_singal_fmri_ts(csv_path)
                    # time_series = time_series[:,:100]
                    time_series = time_series.T

                    # conn = cal_pearson_conn(time_series)
                    # conn = get_correlation_matrices(time_series, thres=0.5)
                    conn = calculate_pearson_correlation(time_series)

                    if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                        nan_list.append(id_indiv)
                    else:
                        try:
                            id_real = id_indiv.split('-')[-1]
                            label_list.append(dict_labels_id[id_real])
                            id_list.append(id_real)
                            conn_list.append(conn)
                        except KeyError:
                            print(id_indiv, 'not in')


    return id_list, time_series_list, conn_list, site_name_list, label_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/ADNI_GO_2/rsfMRI/ROISignals'

    sub_name_list, time_series_list, cor_list, site_name_list, label_list = get_fmri_data(file_dir)

    id_mat = np.array(sub_name_list)
    # time_series_mat = np.array(time_series_list)
    conn_mat = np.array(cor_list).astype(np.float32)
    label_mat = np.array(label_list)


    id_conn_dict = {"id": id_mat, "conn": conn_mat, 'label': label_mat}


    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_adni2.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_adni2_aal116.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_adni2_ad.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_adni2_mci.npy', id_conn_dict)


    print('done')


