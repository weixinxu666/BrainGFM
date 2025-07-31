import numpy as np

# 假设你的数据
data_dict = {
    "type1": np.random.rand(100, 100, 100),  # (batch=100, node=100, dim=100)
    "type2": np.random.rand(50, 333, 333),  # (batch=50, node=333, dim=333)
    "type3": np.random.rand(150, 256, 256),  # (batch=150, node=256, dim=256)
}

# 计算最大 node 和 dimension
max_nodes = max(d.shape[1] for d in data_dict.values())
max_dim = max(d.shape[2] for d in data_dict.values())

# 进行 Padding
padded_data = {}
masks = {}

for key, data in data_dict.items():
    batch_size, node, dim = data.shape
    pad = np.zeros((batch_size, max_nodes, max_dim))  # 统一形状
    mask = np.zeros((batch_size, max_nodes))  # 记录真实数据部分

    # 复制数据
    pad[:, :node, :dim] = data
    mask[:, :node] = 1  # 真实数据部分标记为 1

    padded_data[key] = pad
    masks[key] = mask


