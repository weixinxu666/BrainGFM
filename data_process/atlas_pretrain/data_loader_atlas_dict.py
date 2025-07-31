import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AtlasDataset(Dataset):
    def __init__(self, atlas_dict, batch_size=10, dtype=np.float32):
        """
        初始化 AtlasDataset
        :param atlas_dict: 字典，包含 atlas 名称和对应的文件路径列表
        :param batch_size: 每个批次加载的文件数量
        :param dtype: 数据类型，默认是 np.float32
        """
        self.atlas_dict = atlas_dict  # 包含不同 atlas 的字典
        self.batch_size = batch_size
        self.dtype = dtype
        self.atlas_names = list(atlas_dict.keys())  # 获取所有 atlas 名称
        self.current_atlas_idx = 0  # 当前处理的 atlas 索引

    def __len__(self):
        """返回总的数据集大小"""
        # 计算每个 atlas 的批次数量并求和
        total_batches = 0
        for files in self.atlas_dict.values():
            total_batches += len(files) // self.batch_size + (1 if len(files) % self.batch_size != 0 else 0)
        return total_batches

    def __getitem__(self, idx):
        """按批次返回数据"""
        # 确定当前加载哪个 atlas
        atlas_name = self.atlas_names[self.current_atlas_idx]

        # 获取该 atlas 对应的文件
        files = self.atlas_dict[atlas_name]

        # 计算当前批次的文件范围
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        # 加载当前批次的数据
        batch_data = []
        for file in batch_files:
            data = np.load(file, mmap_mode='r').astype(self.dtype)  # 使用内存映射
            batch_data.append(data)

        # 合并并返回
        batch_data = np.concatenate(batch_data, axis=0)

        # 如果需要可以加入对 atlas_name 的返回
        return torch.tensor(batch_data), atlas_name  # 返回数据和对应的 atlas 名称

    def set_next_atlas(self):
        """切换到下一个 atlas"""
        self.current_atlas_idx = (self.current_atlas_idx + 1) % len(self.atlas_names)
