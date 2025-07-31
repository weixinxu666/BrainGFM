import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

from sklearn.preprocessing import StandardScaler


def calculate_pearson_correlation_bi(matrix, threshold=0.5):
    """
    计算输入矩阵的脑区之间的 Pearson 相关系数矩阵，并进行阈值化和二值化。

    参数:
    matrix (np.ndarray): 形状为 (n_regions, n_timepoints) 的矩阵，
                         其中 n_regions 是脑区数量，n_timepoints 是时间序列长度。
    threshold (float): 二值化的阈值，默认值为 0.5。

    返回:
    np.ndarray: 形状为 (n_regions, n_regions) 的二值化相关系数矩阵。
    """
    # 创建 ConnectivityMeasure 对象，指定 kind='correlation' 使用 Pearson 相关系数
    correlation_measure = ConnectivityMeasure(kind='correlation')

    # 输入矩阵进行计算，结果为一个相关系数矩阵
    correlation_matrix = correlation_measure.fit_transform([matrix.T])[0]

    # 对相关系数矩阵进行阈值化和二值化
    binary_matrix = (correlation_matrix >= threshold).astype(int)

    return binary_matrix

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

def has_no_nan(arr):
    return not np.any(np.isnan(arr))



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


def is_number(s):
    try:
        float(s)  # 如果你只关心整数，可以用 int(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':

    path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABIDE/ABIDE_I_rsfMRI_ROIsignals_Schaefer100ROIs.mat'

    label_path = '/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ABIDE/ABIDE_I.csv'

    mat_data = loadmat(path)
    id_data = mat_data['subID']
    roi_ts_data = mat_data['ROIsignals']

    labels = pd.read_csv(label_path)

    id_list = []
    conn_list = []
    nan_list = []
    label_adhd = []
    # id_conn_dict = {}

    label_id = labels["SUB_ID"]
    label_dx = labels["DX_GROUP"]
    id_dx_dict = {}

    for j in tqdm(range(len(label_dx))):
        id = label_id[j]

        dx = label_dx[j]

        id = '00' + str(id)

        if is_number(dx):

            id_dx_dict.update({str(id): int(dx)})


    for i in tqdm(range(len(id_data))):
    # for i in tqdm(range(5)):
        ids = id_data[i][0][0]

        id_indiv = str(ids).split('_')[0] + '_' + str(ids).split('_')[3]

        roi_ts = roi_ts_data[i][0]
        roi_ts = roi_ts.T

        # roi_ts = normalize_time_series(roi_ts)

        # conn = cal_pearson_conn(roi_ts)
        # conn = get_correlation_matrices(roi_ts, 0.3)
        conn = calculate_pearson_correlation(roi_ts)
        # conn = calculate_pearson_correlation_bi(roi_ts)


        # id_list.append(id_indiv)
        # conn_list.append(conn)

        if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
            nan_list.append(id_indiv)
        else:

            try:
                id_query = id_indiv.split('_')[0]
                label_adhd.append(id_dx_dict[id_query])
                id_list.append(id_indiv)
                conn_list.append(conn)

            except KeyError:
                print(id_query, 'not in dict')

        # id_conn_dict.update({str(id_indiv) : conn})

        print(ids)
        print(roi_ts.shape)

    id_mat = np.array(id_list)
    conn_mat = np.array(conn_list).astype(np.float32)

    label_adhd = [0 if x != 2 else 2 for x in label_adhd]
    label_adhd = [1 if x != 0 else 0 for x in label_adhd]
    label_dx_mat = np.array(label_adhd)


    id_conn_dict = {"id": id_mat, "conn": conn_mat, 'label': label_dx_mat}

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/id_adhd200.npy', id_mat)
    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_abide1.npy', id_conn_dict)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/dict_adhd200.npy', id_conn_dict)

    print('ok')


