import scipy.io
import numpy as np

def get_correlation_matrices(time_series, thres):
    correlation_matrices = np.corrcoef(time_series.T) #T将数组进行转置，转成(n_rois, n_timepoints)
    # threshold = thres  # 阈值，用于过滤弱连接
    # adj_matrix = np.zeros_like(correlation_matrices)
    # adj_matrix[correlation_matrices >= threshold] = 1
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = np.expand_dims(adj_matrix, axis=0)
    return correlation_matrices  # 100x100



# 读取 .mat 文件
data = scipy.io.loadmat('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data_prep/sprps.mat')

# 提取表格数据
table_data = data['a']

fields = table_data.dtype.names

# 解析所有行的数据并存储为字典
parsed_data = {field: [] for field in fields}

for i in range(table_data.shape[0]):  # 遍历1410行
    row = table_data[i, 0]  # 获取第 i 行
    for field in fields:
        # 解包并提取字段数据
        parsed_data[field].append(row[field])


id_list = []
conn_list = []
roi_list = parsed_data['ROISignals_Schaefer100']

for i in parsed_data['subjectID']:
    id_list.append(str(i[0]))

for i in roi_list:
    conn = get_correlation_matrices(i, 0.3)
    conn_list.append(conn)
    print(conn.shape)

id_mat = np.array(id_list)
conn_mat = np.array(conn_list)

id_conn_dict = {"id": id_mat, "conn": conn_mat}

np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/id_sprps.npy', id_mat)
np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/conn_sprps.npy', conn_mat)
np.save('/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data/dict_sprps.npy', id_conn_dict)

print('ok')
