import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure

# from plot_figure.plot_pie import plot_prop
import matplotlib.pyplot as plt

from adjustText import adjust_text
import os


def plot_prop(data_dict):
    # 提取字典的键和值
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())

    # 定义自动生成的配色方案
    cmap = plt.get_cmap("tab20c")  # 使用tab20c配色方案
    colors = cmap(np.linspace(0., 1., len(labels)))

    # 分离每个扇区，较大的扇区稍微分离，较小的扇区不分离
    explode = [0.05 if size >= 1000 else 0 for size in sizes]

    # 创建饼图
    plt.figure(figsize=(12, 12))  # 设置更大的图形
    wedges, texts, autotexts = plt.pie(
        sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12},
    )

    # 隐藏占比较小的扇区标签
    threshold = 1.0  # 只显示大于1.0%的标签
    for i, label in enumerate(labels):
        if sizes[i] / sum(sizes) * 100 < threshold:
            texts[i].set_text('')  # 隐藏小于阈值的标签

    # 调整autotext字体大小
    for autotext in autotexts:
        autotext.set_fontsize(10)

    # 使用 adjustText 自动调整文本标签的位置
    texts_to_adjust = [text for text in texts if text.get_text()] + [autotext for autotext in autotexts if autotext.get_text()]
    adjust_text(texts_to_adjust, only_move={'points':'y', 'text':'y', 'objects':'y'}, force_text=0.5)

    # 添加标题并调整其位置
    plt.title('Distribution of fMRI Data for Pre-training', fontsize=20, y=1.05)
    # plt.title('Distribution of Augmented fMRI Data for Pre-training', fontsize=20, y=1.05)

    # 设置饼图为圆形
    plt.axis('equal')  # 确保饼图是圆形的

    # plt.savefig(os.path.join(save_path + 'data_dist_aug.png'))
    plt.savefig('pre.pdf')
    # plt.savefig('aug.pdf')

    # 显示图形
    # plt.show()   # 有bug


def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)

    # 防止除以零的情况（当 max_val == min_val 时）
    if max_val == min_val:
        return [0 for _ in input_list]  # 所有值都相等时返回全0列表

    # 使用 min-max 归一化
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

    return normalized_list



def get_fmri_data(file_dir):
    data = []
    name = []
    num = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.startswith("conn"):
            # if file.startswith("aug_conn"):
                print(file)
                name.append(file.split('_')[-1].split('.')[0])
                conn = np.load(os.path.join(file_dir, file))
                print(conn.shape)
                data.append(conn)
                num.append(conn.shape[0])

    return data, name, num



if __name__ == '__main__':
    path = '/home/xinxu/Lehigh/Codes/BICLab_data/data'
    save_path = '/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/plot_figure'

    data, name, num = get_fmri_data(path)
    data = np.concatenate(data, axis=0)
    data = data.astype(np.float32)


    print(data.shape)
    print(name)
    print(num)

    data_dict = dict(zip(name, num))

    data_dict = {key.upper(): value for key, value in data_dict.items()}

    plot_prop(data_dict)

    # num = normalize_list(num)

    # np.save('/home/xinxu/Lehigh/Codes/BICLab_data/data/all_aug100.npy', data)

    print('ok')
