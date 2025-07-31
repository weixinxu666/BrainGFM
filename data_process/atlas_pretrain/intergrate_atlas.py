import os
import numpy as np
from collections import defaultdict




# 设定文件夹路径
folder_path = "/home/xinxu/Lehigh/Codes/BICLab_data/atlas_data"  # 修改为你的文件夹路径

# 获取所有 .npy 文件
files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# 按照 atlas 名字分类
atlas_dict = defaultdict(list)

for file in files:
    parts = file.split("_")  # 拆分文件名
    print(parts)
    atlas_name = "_".join(parts[-1:]).replace(".npy", "")  # 提取 atlas 名称
    if atlas_name == 'AAL3v1':
        atlas_dict[atlas_name].append(os.path.join(folder_path, file))

# 存储合并后的数据
combined_dict = {}

# 遍历分类后的 atlas 并合并数据
for atlas_name, file_list in atlas_dict.items():
    combined_data = []

    for file_path in file_list:
        data = np.load(file_path)  # 加载 .npy 文件
        print(data.shape)
        combined_data.append(data.astype('float32'))

    # 沿着 batch 维度 (axis=0) 进行拼接
    combined_array = np.concatenate(combined_data, axis=0)

    # 存入字典
    combined_dict[atlas_name] = combined_array

    print(f"Atlas '{atlas_name}' combined: shape = {combined_array.shape}")

print("所有 atlas 分类并合并完毕！")

# 可选择保存整个字典
# np.save(os.path.join(folder_path, "combined_atlas_dict.npy"), combined_dict)

# 也可以单独保存每个 atlas
for atlas_name, combined_array in combined_dict.items():
    save_path = os.path.join(folder_path, f"combined_{atlas_name}.npy")
    # np.save(save_path, combined_array)
    print(f"Saved: {save_path}")
