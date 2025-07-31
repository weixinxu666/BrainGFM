import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler

from aug_slice import slice_matrix


def has_no_nan(arr):
    return not np.any(np.isnan(arr))


def calculate_pearson_correlation(matrix):
    correlation_measure = ConnectivityMeasure(kind='correlation')

    correlation_matrix = correlation_measure.fit_transform([matrix.T])[0]

    return correlation_matrix



def normalize_time_series(time_series):
    scaler = StandardScaler()
    normalized_time_series = scaler.fit_transform(time_series.T).T  # 转置前后进行标准化
    return normalized_time_series



def get_correlation_matrices(time_series, threshold):
    correlation_matrices = np.corrcoef(time_series) #T将数组进行转置，转成(n_rois, n_timepoints)
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



if __name__ == '__main__':

    path = '/home/xinxu/Lehigh/Codes/BICLab_data/ADHD200/ADHD200_rsfMRI_ROISignals_Schaefer100ROIs.mat'

    mat_data = loadmat(path)
    id_data = mat_data['subID']
    roi_ts_data = mat_data['ROIsignals']

    id_list = []
    conn_list = []
    nan_list = []
    # id_conn_dict = {}

    for i in tqdm(range(len(id_data))):
    # for i in tqdm(range(5)):
        ids = id_data[i][0][0]

        id_indiv = str(ids).split('_')[0] + '_' + str(ids).split('_')[3]

        roi_ts = roi_ts_data[i][0]
        roi_ts = roi_ts.T
        # if has_no_nan(roi_ts):

        # roi_ts = normalize_time_series(roi_ts)

        # conn = cal_pearson_conn(roi_ts)

        # conn = calculate_pearson_correlation(roi_ts)

        roi_ts_slice = slice_matrix(roi_ts)
        for slice_i in roi_ts_slice:
            # conn = calculate_pearson_correlation(slice_i)
            conn = get_correlation_matrices(slice_i, 0.3)

            if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                nan_list.append(id_indiv)
            else:
                id_list.append(id_indiv)
                conn_list.append(conn)

            # id_conn_dict.update({str(id_indiv) : conn})

            print(ids)
            print(roi_ts.shape)

    id_mat = np.array(id_list)
    conn_mat = np.array(conn_list).astype(np.float32)


    id_conn_dict = {"id": id_mat, "conn": conn_mat}

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/id_adhd200.npy', id_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_adhd200.npy', conn_mat)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/dict_adhd200.npy', id_conn_dict)

    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_adhd200.npy', conn_mat)

    print('ok')


