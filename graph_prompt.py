# graph_prompt.py
import torch
import torch.nn as nn

class GraphPrompt(nn.Module):
    """
    同时对 Node/Edge 做条件化调制：
    - Node prompt: 对节点特征逐元素缩放（可选再加偏置）
    - Edge prompt: 产生边的加法偏置（logit级），与原图融合得到 A_tilde
    条件输入：disease_token、parcellation_token（已投影到 hidden_dim）
    """
    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int,
        max_feature_dim: int,
        node_mode: str = "scale",     # "scale" | "scale_bias"
        edge_strength: float = 0.3,   # edge 注入强度
        init_std: float = 1e-2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.max_feature_dim = max_feature_dim
        self.node_mode = node_mode
        self.edge_strength = edge_strength

        # 基础 prompts（全局可学习）
        self.base_node_prompt = nn.Parameter(torch.randn(1, max_nodes, max_feature_dim) * init_std)
        self.base_edge_prompt = nn.Parameter(torch.randn(1, max_nodes, max_nodes) * init_std)

        # 条件化 MLP：由 (disease, parc) 生成 node/edge 门控参数
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [alpha_node, beta_node, alpha_edge, beta_edge]
        )

    @staticmethod
    def _mask_invalid_nodes_BND(tensor_BND, valid_num_nodes):
        # tensor: [B,N,D]
        B, N, _ = tensor_BND.shape
        device = tensor_BND.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, n in enumerate(valid_num_nodes):
            if n < N:
                mask[i, n:] = True
        return tensor_BND.masked_fill(mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _mask_invalid_nodes_BNN(tensor_BNN, valid_num_nodes):
        # tensor: [B,N,N]
        B, N, _ = tensor_BNN.shape
        device = tensor_BNN.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, n in enumerate(valid_num_nodes):
            if n < N:
                mask[i, n:] = True
        tensor_BNN = tensor_BNN.masked_fill(mask.unsqueeze(-1), 0.0)
        tensor_BNN = tensor_BNN.masked_fill(mask.unsqueeze(1), 0.0)
        return tensor_BNN

    def forward(self, node_feats_BND, adj_BNN, disease_token_B1H, parc_token_B1H, valid_num_nodes):
        """
        Inputs:
            node_feats_BND: [B, N, D']  —— 建议在 projection 之前调用
            adj_BNN:        [B, N, N]
            disease_token_B1H, parc_token_B1H: [B,1,H]
            valid_num_nodes: List[int]，长度为 B
        Returns:
            node_feats_prompted: [B, N, D']
            A_tilde:             [B, N, N]
            attn_bias_nodes:     [B, N, N] —— 可用于 Transformer 注意力偏置（可选）
        """
        B, N, Dp = node_feats_BND.shape

        # 条件门控
        cond = torch.cat([disease_token_B1H.squeeze(1), parc_token_B1H.squeeze(1)], dim=-1)  # [B,2H]
        gate = self.cond_mlp(cond)  # [B,4]
        alpha_node, beta_node, alpha_edge, beta_edge = gate.chunk(4, dim=-1)
        alpha_node = alpha_node.view(B, 1, 1)
        beta_node  = beta_node.view(B, 1, 1)
        alpha_edge = alpha_edge.view(B, 1, 1)
        beta_edge  = beta_edge.view(B, 1, 1)

        # === Node prompt ===
        node_prompt = self.base_node_prompt[:, :N, :Dp].expand(B, -1, -1)       # [B,N,Dp]
        node_prompt = self._mask_invalid_nodes_BND(node_prompt, valid_num_nodes)

        if self.node_mode == "scale":
            scale = torch.tanh(alpha_node * node_prompt + beta_node)            # [-1,1]
            node_feats_prompted = node_feats_BND * (1.0 + scale)
        else:
            scale = torch.tanh(alpha_node * node_prompt + beta_node)
            bias  = torch.tanh(node_prompt)
            node_feats_prompted = node_feats_BND * (1.0 + scale) + 0.1 * bias   # 偏置小一点

        # === Edge prompt ===
        edge_base = self.base_edge_prompt[:, :N, :N].expand(B, -1, -1)          # [B,N,N]
        edge_base = 0.5 * (edge_base + edge_base.transpose(1, 2))               # 对称
        edge_base = edge_base - torch.diag_embed(torch.diagonal(edge_base, dim1=1, dim2=2))  # 零对角
        edge_base = self._mask_invalid_nodes_BNN(edge_base, valid_num_nodes)

        edge_logits = alpha_edge * edge_base + beta_edge
        edge_add = torch.sigmoid(edge_logits)                                    # [B,N,N]∈[0,1]

        Adj_soft = adj_BNN.float()
        A_tilde = torch.clamp(Adj_soft + self.edge_strength * edge_add * (1.0 - Adj_soft), 0.0, 1.0)
        A_tilde = 0.5 * (A_tilde + A_tilde.transpose(1, 2))
        A_tilde = A_tilde - torch.diag_embed(torch.diagonal(A_tilde, dim1=1, dim2=2))

        # Transformer 自注意力的加法偏置（节点-节点）
        attn_bias_nodes = 3.0 * (edge_add - 0.5)  # ~[-1.5, 1.5]

        return node_feats_prompted, A_tilde, attn_bias_nodes


if __name__ == "__main__":

    def build_symmetric_adj(B, N, density=0.2, device="cpu", seed=0):
        """
        生成目标稀疏度的对称邻接（零对角），近似达到 density（仅统计非对角）。
        """
        g = torch.Generator(device=device).manual_seed(seed)
        m = torch.rand(B, N, N, generator=g, device=device)
        m = (m + m.transpose(1, 2)) / 2
        # 选阈值实现给定密度（仅考虑上三角）
        triu_idx = torch.triu_indices(N, N, offset=1)
        flat = m[:, triu_idx[0], triu_idx[1]]  # [B, N*(N-1)/2]
        kth = torch.quantile(flat, 1 - density, dim=1, keepdim=True)
        keep = (flat >= kth).float()
        # 还原为方阵
        adj = torch.zeros(B, N, N, device=device)
        adj[:, triu_idx[0], triu_idx[1]] = keep
        adj = adj + adj.transpose(1, 2)
        # 零对角
        adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))
        return adj

    def offdiag_density(adj):
        """统计非对角稠密度"""
        B, N, _ = adj.shape
        off = adj.sum(dim=(1,2))  # 对角已为0
        return (off / (N * (N - 1))).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)

    # --------- 配置（可改）---------
    B = 3
    N = 64
    Din = 80          # 原始节点特征维度（不超过 max_feature_dim）
    hidden_dim = 128
    max_nodes = 256
    max_feature_dim = 256
    base_density = 0.15  # 原图稀疏度（越小越稀）

    # --------- 随机数据 ---------
    x = torch.randn(B, N, Din, device=device)            # 节点特征
    adj = build_symmetric_adj(B, N, density=base_density, device=device)

    # 有效节点数（测试无效节点屏蔽）
    valid_num_nodes = [N, N - 10, N - 5]                 # 第2/3个样本尾部节点无效
    # 把输入 adj 的无效节点行列也置零，以符合真实预处理
    for i, n in enumerate(valid_num_nodes):
        if n < N:
            adj[i, n:, :] = 0.0
            adj[i, :, n:] = 0.0

    # 条件 token（这里不依赖 ClinicalBERT，直接随机）
    disease_token = torch.randn(B, 1, hidden_dim, device=device)
    parc_token    = torch.randn(B, 1, hidden_dim, device=device)

    # --------- 实例化 GraphPrompt ---------
    gp = GraphPrompt(
        hidden_dim=hidden_dim,
        max_nodes=max_nodes,
        max_feature_dim=max_feature_dim,
        node_mode="scale",        # 或 "scale_bias"
        edge_strength=0.3,        # 注入强度
        init_std=1e-2
    ).to(device)
    gp.eval()

    # --------- 前向测试 ---------
    with torch.no_grad():
        x_prompted, A_tilde, attn_bias = gp(
            node_feats_BND=x,            # [B,N,Din]
            adj_BNN=adj,                 # [B,N,N]
            disease_token_B1H=disease_token,   # [B,1,H]
            parc_token_B1H=parc_token,         # [B,1,H]
            valid_num_nodes=valid_num_nodes
        )

    # --------- 打印与校验 ---------
    print("== Shapes ==")
    print("x:", x.shape, " -> x_prompted:", x_prompted.shape)
    print("adj:", adj.shape, " -> A_tilde:", A_tilde.shape)
    print("attn_bias:", attn_bias.shape)

    # 对称性与零对角
    sym_err = (A_tilde - A_tilde.transpose(1, 2)).abs().max().item()
    diag_max = torch.diagonal(A_tilde, dim1=1, dim2=2).abs().max().item()
    print(f"symmetry max error: {sym_err:.4e}, zero-diagonal max: {diag_max:.4e}")

    # 稠密度变化（非对角）
    dens_before = offdiag_density(adj)
    dens_after  = offdiag_density(A_tilde)
    print("off-diagonal density before:", [round(d, 4) for d in dens_before])
    print("off-diagonal density after :", [round(d, 4) for d in dens_after])

    # 无效节点屏蔽检查：A_tilde 的无效区域应为 0
    invalid_ok = True
    for i, n in enumerate(valid_num_nodes):
        if n < N:
            block = A_tilde[i, n:, :]
            block2= A_tilde[i, :, n:]
            if block.abs().max().item() > 1e-6 or block2.abs().max().item() > 1e-6:
                invalid_ok = False
                break
    print("invalid-node masking OK?:", invalid_ok)

    # 节点特征调制强度（均方差）
    delta = (x_prompted - x).pow(2).mean(dim=(1,2)).sqrt().tolist()
    print("RMS(node delta) per graph:", [round(d, 6) for d in delta])

    print("\nDone.")