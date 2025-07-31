import numpy as np
from scipy.stats import pearsonr
# from scipy.io import loadmat
from tqdm import tqdm
from mat73 import loadmat
from nilearn.connectome import ConnectivityMeasure

from aug_slice import slice_matrix







def get_correlation_matrices(time_series):
    correlation_matrices = np.corrcoef(time_series) #T将数组进行转置，转成(n_rois, n_timepoints)
    return correlation_matrices  # 100x100


def cal_pearson_conn(time_series):

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

    return connectivity_matrix



if __name__ == '__main__':

    # id_path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABCD/rsfMRI_ABCD_subjectID.mat'
    # path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABCD/rsfMRI_Schaefer135ROIs_ROISignals_ABCD.mat'
    path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABCD/rsfMRI_Schaefer200ROIs_ROISignals_ABCD.mat'
    print('loading')
    mat_data = loadmat(path)

    id_data = mat_data['subjectID']
    roi_ts_data = mat_data['ROISignals']

    id_list = []
    conn_list = []
    nan_list = []
    # id_conn_dict = {}

    print('loading finished')

    for i in tqdm(range(len(id_data))):
    # for i in tqdm(range(5)):
        ids = id_data[i]

        id_indiv = str(ids).split('_')[0] + '_' +str(ids).split('_')[2]

        roi_ts = roi_ts_data[i][0][:,:200]
        roi_ts = roi_ts.T

        roi_ts_slice = slice_matrix(roi_ts)

        for slice_i in roi_ts_slice:

            # conn = calculate_pearson_correlation(slice_i)

            # conn = cal_pearson_conn(roi_ts)
            conn = get_correlation_matrices(slice_i)
            # conn = get_correlation_matrices(roi_ts)


            if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                nan_list.append(id_indiv)
                print('*********************************************')
            else:
                id_list.append(id_indiv)
                conn_list.append(conn)


                print(ids)
                print(roi_ts.shape)

    id_mat = np.array(id_list)
    conn_mat = np.array(conn_list)

    print(nan_list)


    id_conn_dict = {"id": id_mat, "conn": conn_mat}

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/id_abcd.npy', id_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_abcd.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/dict_abcd.npy', id_conn_dict)

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_abcd_sf200.npy', conn_mat)
    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_abcd_sf200.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_abcd.npy', conn_mat)

    print('ok')


