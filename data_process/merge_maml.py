import os
import numpy as np

# 设置.npy文件所在目录
npy_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/downstream_maml'
merged_data = {}

for fname in os.listdir(npy_dir):
    if not fname.endswith('.npy'):
        continue

    file_path = os.path.join(npy_dir, fname)
    key_name = os.path.splitext(fname)[0].split('dict_')[-1]

    try:
        data_dict = np.load(file_path, allow_pickle=True).item()

        if not isinstance(data_dict, dict):
            raise ValueError("不是字典结构")
        if len(data_dict) != 3:
            raise ValueError(f"{fname} 中的 key 数量不是3个")

        keys = list(data_dict.keys())

        conn_key = keys[1]
        label_key = keys[2]

        merged_data[key_name] = {
            "conn": data_dict[conn_key],          # 保留 conn
            "label": data_dict[label_key]         # 直接用数组形式存储 label
        }

    except Exception as e:
        print(f"[!] 处理文件 {fname} 出错：{e}")
        continue

np.save("maml_all.npy", merged_data)
print("✅ 合并完成，保存为 maml_all.npy")
