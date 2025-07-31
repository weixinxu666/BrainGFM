import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure

from nilearn.connectome import ConnectivityMeasure

from aug_slice import slice_matrix



def calculate_pearson_correlation(matrix):
    correlation_measure = ConnectivityMeasure(kind='correlation')

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
    num_list = []
    sub_name_list = []
    time_series_list = []
    cor_list = []
    site_name_list = []

    name_list = os.listdir(file_dir)

    # for site_name in tqdm(site_list[:1]):
    for name in tqdm(name_list):

        dir_path = os.path.join(file_dir, name)
        # site_dir_list = os.listdir(site_dir_path)


        sub_name = name.split('_')[0] + '_' + name.split('_')[2]

        num_list.append(sub_name)

        print(sub_name)

        time_series = read_singal_fmri_ts(dir_path)
        time_series = time_series[:,:100]
        time_series = time_series.T
        time_series_list.append(time_series)
        # cor = get_correlation_matrices(time_series, thres=0.5)

        print(time_series.shape)

        roi_ts_slice = slice_matrix(time_series)
        for slice_i in roi_ts_slice:
            # conn = calculate_pearson_correlation(slice_i)
            conn = get_correlation_matrices(slice_i)

            sub_name_list.append(sub_name)
            # print(time_series.shape)
            cor_list.append(conn)


    return sub_name_list, time_series_list, cor_list, site_name_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/Resting_fMRI_Connectivity/EMBARC_rsfMRI_fMRIPrep_Schaefer100_tp1_Baseline'

    sub_name_list, time_series_list, cor_list, site_name_list = get_fmri_data(file_dir)

    id_mat = np.array(sub_name_list)
    # time_series_mat = np.array(time_series_list)
    conn_mat = np.array(cor_list).astype(np.float32)

    # print(conn_mat.shape)


    id_conn_dict = {"id": id_mat, "conn": conn_mat}

    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/id_embarc.npy', id_mat)
    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/conn_embarc.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/dict_embarc.npy', id_conn_dict)

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_embarc.npy', conn_mat)


    print('done')


