import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
from tqdm import tqdm
import math

from nilearn.connectome import ConnectivityMeasure

import pandas as pd



def calculate_pearson_correlation(matrix):
    correlation_measure = ConnectivityMeasure(kind='correlation')

    correlation_matrix = correlation_measure.fit_transform([matrix.T])[0]

    return correlation_matrix



def get_correlation_matrices(time_series):
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
    aug = False

    path = '/home/xinxu/Lehigh/Codes/BICLab_data/OASIS-3/fMRIdata_OASIS3_Schaefer100.mat'
    # path = '/home/xinxu/Lehigh/Codes/BICLab_data/OASIS-3/fMRIdata_OASIS3_Schaefer200.mat'

    label_path = '/home/xinxu/Lehigh/Codes/BICLab_data/OASIS-3/PhenotypicData/OASIS-3_ADRC Clinical Data.csv'



    labels = pd.read_csv(label_path)

    mat_data = loadmat(path)
    id_data = mat_data['ROISignals_Schaefer100'][:,1]
    roi_ts_data = mat_data['ROISignals_Schaefer100'][:,0]

    id_list = []
    id_ind_list = []
    conn_list = []
    nan_list = []
    label_list= []
    # id_conn_dict = {}

    label_id = labels["ADRC_ADRCCLINICALDATA ID"]
    label_dx = labels["dx1"]
    id_dx_dict = {}

    for j in tqdm(range(len(label_dx))):
        id = label_id[j].replace("_ClinicalData_", "_")
        # print(j)
        print(id)
        id_list.append(id)

        dx = label_dx[j]

        if isinstance(dx, str):  # 确保 dx 是字符串
            if dx.lower() not in ["nan", ".", "none", ""]:  # 过滤掉无效字符串
                dx = 0 if "normal" in dx.lower() else 1
                id_dx_dict[str(id)] = dx  # 直接赋值，不用 update()
        elif isinstance(dx, float) and not math.isnan(dx):  # 确保 dx 不是 NaN
            id_dx_dict[str(id)] = int(dx)  # 只有 dx 不是 NaN 时才转换

        if is_number(dx) and not math.isnan(dx):

            id_dx_dict.update({str(id): int(dx)})

    # print(id_dx_dict)

    for i in tqdm(range(len(id_data))):

        id_indiv = id_data[i][0].split('_task')[0]
        id_ind_list.append(id_indiv)
        # print(i)
        # print(id_indiv)

        roi_ts = roi_ts_data[i]
        roi_ts = roi_ts.T

        # conn = cal_pearson_conn(roi_ts)
        # conn = get_correlation_matrices(roi_ts, 0.3)

        # id_list.append(ids)
        # conn_list.append(conn)



        conn = get_correlation_matrices(roi_ts)

        # if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
        #     nan_list.append(id_indiv)
        # else:
        #     try:
        #         # label_list.append(id_dx_dict[id_indiv])
        #         id_list.append(id_indiv)
        #         conn_list.append(conn)
        #     except KeyError:
        #         print('not')

        print(id_list)







        # id_conn_dict.update({str(id_indiv) : conn})

        # print(id_indiv)
        # print(roi_ts.shape)

    intersection = set(id_list) & set(id_ind_list)

    # 输出交集元素及数量
    print("Intersection:", intersection)
    print("Number of common elements:", len(intersection))

    id_mat = np.array(id_list)
    conn_mat = np.array(conn_list).astype(np.float32)
    label_mat = np.array(label_list)


    id_conn_dict = {"id": id_mat, "conn": conn_mat, 'label': label_mat}


    np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_oasis.npy', id_conn_dict)

    # print(conn_mat.shape)


    # id_conn_dict = {"id": id_mat, "conn": conn_mat, "label": }

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/downstream/label_dict_oasis.npy', id_conn_dict)

    print('ok')


