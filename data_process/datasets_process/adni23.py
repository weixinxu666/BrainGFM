#
# import numpy as np
# from scipy.stats import pearsonr
# from scipy.io import loadmat
# from tqdm import tqdm
#
#
#
# def get_correlation_matrices(time_series, threshold):
#     correlation_matrices = np.corrcoef(time_series) #T将数组进行转置，转成(n_rois, n_timepoints)
#     # adj_matrix = np.zeros_like(correlation_matrices)
#     # adj_matrix[correlation_matrices >= threshold] = 1
#     # adj_matrix = np.triu(adj_matrix)
#     # adj_matrix = np.expand_dims(adj_matrix, axis=0)
#     return correlation_matrices  # 100x100
#
#
# def cal_pearson_conn(time_series):
#     # input is :  regions x time point
#     n_regions = time_series.shape[0]
#     connectivity_matrix = np.zeros((n_regions, n_regions))
#
#     # 计算Pearson相关系数
#     for i in range(n_regions):
#         for j in range(i, n_regions):
#             if i == j:
#                 connectivity_matrix[i, j] = 1  # 自相关为1
#             else:
#                 # 计算Pearson相关系数
#                 corr, _ = pearsonr(time_series[i], time_series[j])
#                 connectivity_matrix[i, j] = corr
#                 connectivity_matrix[j, i] = corr  # 矩阵对称
#
#     # connectivity_matrix 是最终的功能连接矩阵
#     # print(connectivity_matrix)
#     return connectivity_matrix
#
#
#
# if __name__ == '__main__':
#
#     path = '/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/rsfMRI_ROISignals_phenotypic_ADNI2_ADNI3_20231210.mat'
#
#     mat_data = loadmat(path)
#     id_data = mat_data['subID']
#     roi_ts_data = mat_data['ROIsignals']
#
#     id_list = []
#     conn_list = []
#     # id_conn_dict = {}
#
#     for i in tqdm(range(len(id_data))):
#     # for i in tqdm(range(5)):
#         ids = id_data[i][0][0]
#
#         id_indiv = str(ids).split('_')[0] + '_' + str(ids).split('_')[3]
#
#         roi_ts = roi_ts_data[i][0]
#         roi_ts = roi_ts.T
#
#         # conn = cal_pearson_conn(roi_ts)
#         conn = get_correlation_matrices(roi_ts, 0.3)
#
#         id_list.append(id_indiv)
#         conn_list.append(conn)
#
#         # id_conn_dict.update({str(id_indiv) : conn})
#
#         print(ids)
#         print(roi_ts.shape)
#
#     id_mat = np.array(id_list)
#     conn_mat = np.array(conn_list)
#
#
#     id_conn_dict = {"id": id_mat, "conn": conn_mat}
#
#     np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/id_adni23.npy', id_mat)
#     np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/conn_adni23.npy', conn_mat)
#
#     np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/dict_adni23.npy', id_conn_dict)
#
#     print('ok')
#
#
