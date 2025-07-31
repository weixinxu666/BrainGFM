import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
import warnings

warnings.filterwarnings("ignore")


def read_singal_fmri_ts(data_file):
    """
    读取 fMRI 时间序列 CSV 文件。
    """
    try:
        # 尝试读取文件，使用 latin1 编码解决编码问题
        csv_data = pd.read_csv(data_file, header=None, encoding='latin1')
        data = np.array(csv_data)
        roi_ts = []
        for d in data:
            ts = d[0].split(' ')
            roi_ts.append(ts)
        return np.array(roi_ts).astype('float64')
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: Failed to read {data_file}. Please check the file encoding.")
        return None  # 返回空值，继续处理其他文件
    except Exception as e:
        print(f"Error: {e} occurred while reading {data_file}.")
        return None


def get_fmri_data(file_dir):
    """
    处理 fMRI 数据目录，提取时间序列和相关矩阵。
    """
    id_list = []
    conn_list = []
    nan_list = []

    atlas_data = {}

    name_list = os.listdir(file_dir)

    for name in tqdm(name_list):
        dir_path = os.path.join(file_dir, name)
        if name == ".DS_Store" or name.startswith("._"):
            continue

        files = os.listdir(dir_path)
        for file_name in files:
            if not file_name.endswith(".csv"):
                continue  # 跳过非 CSV 文件

            print(file_name)

            id_indiv = file_name.split('sub-')[-1].split('_task')[0]
            csv_path = os.path.join(dir_path, file_name)
            time_series = read_singal_fmri_ts(csv_path)

            if time_series is None:  # 跳过读取失败的文件
                nan_list.append(id_indiv)
                continue

            time_series = time_series.T

            try:
                conn = np.corrcoef(time_series)
                if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                    nan_list.append(id_indiv)
                else:
                    id_list.append(id_indiv)
                    conn_list.append(conn)
            except Exception as e:
                print(f"Error while processing {file_name}: {e}")
                nan_list.append(id_indiv)

    return id_list, conn_list, nan_list


if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/ABIDE/ABIDE_I_ROISignals'

    id_list, conn_list, nan_list = get_fmri_data(file_dir)

    conn_mat = np.array(conn_list).astype(np.float32)

    print(f"Processed {len(conn_list)} files successfully.")
    print(f"Skipped {len(nan_list)} files due to errors.")
    print(conn_mat.shape)

    # 保存结果
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/id_list.npy', id_list)
    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_mat.npy', conn_mat)

    print('done')
