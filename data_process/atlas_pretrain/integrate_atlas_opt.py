import os
import numpy as np
from collections import defaultdict

# 设定文件夹路径
folder_path = "/home/xinxu/Lehigh/Codes/BICLab_data/atlas_data"  # 修改为你的文件夹路径
save_file_path = "/home/xinxu/Lehigh/Codes/BICLab_data/atlas_group"  # 修改为你的文件夹路径

# 获取所有 .npy 文件
files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# 按照 atlas 名字分类
atlas_dict = defaultdict(list)

for file in files:
    parts = file.split("_")  # 拆分文件名
    print(parts)
    atlas_name = "_".join(parts[-1:]).replace(".npy", "")  # 提取 atlas 名称
    if atlas_name != 'AAL3v1':
        atlas_dict[atlas_name].append(os.path.join(folder_path, file))

# 存储合并后的数据
combined_dict = {}

# 遍历分类后的 atlas 并逐个处理数据
for atlas_name, file_list in atlas_dict.items():
    combined_data = []

    for file_path in file_list:
        # 使用内存映射加载数据
        data = np.load(file_path, mmap_mode='r')  # 使用内存映射方式加载
        combined_data.append(data.astype('float32'))

        # 为避免内存问题，可以选择每次合并后直接保存中间结果
        # 例如：合并后的每个小批次可以保存为单独的文件
        # 每次拼接后保存数据
        if len(combined_data) >= 10:  # 每次合并10个文件后保存
            combined_array = np.concatenate(combined_data, axis=0)
            save_path = os.path.join(save_file_path, f"combined_{atlas_name}.npy")
            np.save(save_path, combined_array)
            print(f"Partial Save: {save_path}")
            combined_data = []  # 清空合并数据

    # 处理剩余的文件
    # if combined_data:
    #     combined_array = np.concatenate(combined_data, axis=0)
    #     save_path = os.path.join(save_file_path, f"combined_{atlas_name}_final.npy")
    #     # np.save(save_path, combined_array)
    #     print(f"Final Save: {save_path}")

print("所有 atlas 分类并合并完毕！")
