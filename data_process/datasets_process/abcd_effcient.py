import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from mat73 import loadmat
from nilearn.connectome import ConnectivityMeasure
from aug_slice import slice_matrix

def get_correlation_matrices(time_series):
    correlation_matrices = np.corrcoef(time_series)  # 转置为 (n_rois, n_timepoints)
    return correlation_matrices  # 返回相关矩阵

if __name__ == '__main__':
    # 路径设置
    path = '/home/xinxu/Lehigh/Codes/BICLab_data/ABCD/rsfMRI_Schaefer200ROIs_ROISignals_ABCD.mat'
    output_path = '/home/xinxu/Lehigh/Codes/BICLab_data/data/aug_conn_abcd_sf200.npy'

    print('loading')
    mat_data = loadmat(path)
    id_data = mat_data['subjectID']
    roi_ts_data = mat_data['ROISignals']

    nan_list = []
    chunk_size = 100  # 每次处理的块大小
    conn_shape = (200, 200)  # ROI 矩阵的形状
    total_subjects = len(id_data)

    print('loading finished')

    with open(output_path, 'wb') as f_out:
        for start_idx in tqdm(range(0, total_subjects, chunk_size)):
            # 分块加载数据
            end_idx = min(start_idx + chunk_size, total_subjects)
            id_list = []
            conn_list = []

            for i in range(start_idx, end_idx):
                ids = id_data[i]
                id_indiv = str(ids).split('_')[0] + '_' + str(ids).split('_')[2]

                roi_ts = roi_ts_data[i][0][:, :200].T
                roi_ts_slice = slice_matrix(roi_ts)

                for slice_i in roi_ts_slice:
                    conn = get_correlation_matrices(slice_i)

                    if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                        nan_list.append(id_indiv)
                    else:
                        id_list.append(id_indiv)
                        conn_list.append(conn)

            # 保存当前分块数据到文件
            np.save(f_out, np.array(conn_list))  # 保存连接矩阵
            print(f"Processed subjects {start_idx} to {end_idx}")

    print("Finished processing. Nan list:", nan_list)
