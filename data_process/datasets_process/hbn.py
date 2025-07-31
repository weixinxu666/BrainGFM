import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure


import warnings
# warnings.filterwarnings("ignore")

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
    sub_name_list = []
    time_series_list = []
    cor_list = []
    site_name_list = []

    id_list = []
    conn_list = []
    nan_list = []

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
                # conn = get_correlation_matrices(time_series, thres=0.5)

                # sub_name_list.append(sub_name)
                # print(time_series.shape)
                # time_series_list.append(time_series)
                # # print(time_series_100.shape)
                # cor_list.append(cor)
                # site_name_list.append(site_name)

                roi_ts_slice = slice_matrix(time_series)
                for slice_i in roi_ts_slice:
                    # conn = calculate_pearson_correlation(slice_i)
                    conn = get_correlation_matrices(slice_i)

                    if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                        nan_list.append(id_indiv)
                    else:
                        id_list.append(id_indiv)
                        conn_list.append(conn)

    return id_list, time_series_list, conn_list, site_name_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/rsfMRI/rsfMRI_Preprocessed_ROISignals_Schaefer135ROIs_AlexZhao'

    sub_name_list, time_series_list, cor_list, site_name_list = get_fmri_data(file_dir)

    id_mat = np.array(sub_name_list)
    # time_series_mat = np.array(time_series_list)
    conn_mat = np.array(cor_list).astype(np.float32)

    print(conn_mat.shape)


    id_conn_dict = {"id": id_mat, "conn": conn_mat}

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/id_hbn.npy', id_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_hbn.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/dict_hbn.npy', id_conn_dict)

    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_hbn.npy', conn_mat)


    print('done')


