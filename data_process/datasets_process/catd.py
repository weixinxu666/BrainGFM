import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure


import warnings
warnings.filterwarnings("ignore")

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

    name_list = os.listdir(file_dir)

    # for site_name in tqdm(site_list[:1]):
    for csv_file in tqdm(name_list):
        sub_name = csv_file.split('_')[0] + csv_file.split('_')[1]
        csv_path = os.path.join(file_dir, csv_file)
        time_series = read_singal_fmri_ts(csv_path)
        time_series = time_series[:,:100]
        time_series = time_series.T
        # cor = get_correlation_matrices(time_series, thres=0.5)

        roi_ts_slice = slice_matrix(time_series)
        for slice_i in roi_ts_slice:
            # conn = calculate_pearson_correlation(slice_i)
            conn = get_correlation_matrices(slice_i)

            sub_name_list.append(sub_name)
            print(time_series.shape)
            cor_list.append(conn)


    return sub_name_list, time_series_list, cor_list, site_name_list





if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/CATD/TimeSeries/schaefer100_subcortex'

    sub_name_list, time_series_list, cor_list, site_name_list = get_fmri_data(file_dir)

    id_mat = np.array(sub_name_list)
    # time_series_mat = np.array(time_series_list)
    conn_mat = np.array(cor_list).astype(np.float32)

    print(conn_mat.shape)


    id_conn_dict = {"id": id_mat, "conn": conn_mat}

    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/id_catd.npy', id_mat)
    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/conn_catd.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/dict_catd.npy', id_conn_dict)

    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_catd.npy', conn_mat)


    print('done')


