# braingfm.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from disease_names import disease_names

# 导入 GraphPrompt
from graph_prompt import GraphPrompt

# ===========================================
# ClinicalBERT 提取疾病嵌入（带 fallback）
# ===========================================
def get_disease_embeddings(embed_dim=768, seed=42):
    """
    优先用 ClinicalBERT 提取疾病名称 CLS；失败时回退随机。
    返回：dict[str, torch.Tensor]，value 形状为 [768]
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model.eval()
        embeddings = {}
        with torch.no_grad():
            for disease, name in disease_names.items():
                inputs = tokenizer(name, return_tensors='pt')
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1,768]
                embeddings[disease.lower()] = cls_embedding.squeeze(0).cpu()
        return embeddings
    except Exception:
        g = torch.Generator().manual_seed(seed)
        embeddings = {}
        for disease in disease_names.keys():
            embeddings[disease.lower()] = torch.randn(embed_dim, generator=g)
        return embeddings

# ===========================================
# FastMoE: FFN 与 GCN
# ===========================================
class FastMoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [B,N,H]
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B,E]
        top1 = torch.argmax(scores, dim=-1)  # [B]
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx])
        return out

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.act = act
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        # x: [B,N,Fin], adj: [B,N,N]
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.size(0), -1, -1))  # [B,N,Fout]
        out = torch.bmm(adj, support)  # [B,N,Fout]
        if self.bias is not None:
            out = out + self.bias
        out = self.bn(out.view(-1, out.shape[-1])).view(out.shape)
        return self.act(out)

class FastMoEGCN(nn.Module):
    def __init__(self, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([GraphConvolution(hidden_dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x, adj):
        # x: [B,N,H], adj: [B,N,N]
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B,E]
        top1 = torch.argmax(scores, dim=-1)  # [B]
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx], adj[idx])
        return out

# ===========================================
# UGFormer Encoder Layer
# ===========================================
class GTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FastMoEFFN(d_model, dim_feedforward, num_experts, dropout)

    def forward(self, src, src_mask=None, is_causal=None, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src

# ===========================================
# BrainGFM（集成 GraphPrompt）
# ===========================================
class BrainGFM(nn.Module):
    def __init__(self, ff_hidden_size, num_classes, num_self_att_layers, dropout, num_GNN_layers, nhead,
                 hidden_dim=128, max_feature_dim=256, rwse_steps=5, max_nodes=256, moe_num_experts=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_feature_dim = max_feature_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes

        # 输入特征 -> hidden_dim
        self.projection_layer = nn.Linear(self.max_feature_dim, self.hidden_dim)
        self.disease_proj = nn.Linear(768, self.hidden_dim)

        # 图谱 tokens（原始维度，进 projection）
        self.parcellation_tokens = nn.ParameterDict({
            'schaefer':   nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer200':nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer300':nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'shen268':    nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'power264':   nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'gordon333':  nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal116':     nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal3v1':     nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
        })

        # 疾病 tokens（注册 Parameter）
        disease_embed_dict = get_disease_embeddings()
        self.disease_embeddings = {}
        for k, v in disease_embed_dict.items():
            param = nn.Parameter(v.unsqueeze(0).unsqueeze(0))  # [1,1,768]
            self.disease_embeddings[k.lower()] = param
            self.register_parameter(f'disease_embedding_{k.lower()}', param)

        # 统一的 GraphPrompt
        self.graph_prompt = GraphPrompt(
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            max_feature_dim=max_feature_dim,
            node_mode="scale",
            edge_strength=0.3,
            init_std=1e-2
        )

        # UGFormer 堆叠
        self.ugformer_layers = nn.ModuleList([
            TransformerEncoder(
                GTransformerEncoderLayer(
                    d_model=hidden_dim, nhead=nhead,
                    dim_feedforward=ff_hidden_size, dropout=dropout, num_experts=moe_num_experts
                ),
                num_layers=num_self_att_layers
            )
            for _ in range(num_GNN_layers)
        ])
        # MoE-GCN 堆叠
        self.lst_gnn = nn.ModuleList([FastMoEGCN(hidden_dim, moe_num_experts) for _ in range(num_GNN_layers)])
        self.predictions = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_GNN_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_GNN_layers)])

    def compute_rwse(self, adj, k):
        # 简单随机游走对角特征
        B, N, _ = adj.shape
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw = adj.clone()
        diag_features = []
        for _ in range(k):
            rw_diag = torch.diagonal(rw, dim1=1, dim2=2).unsqueeze(-1)  # [B,N,1]
            diag_features.append(rw_diag)
            rw = torch.bmm(rw, adj)
        return torch.cat(diag_features, dim=-1)  # [B,N,k]

    def expand_adj_block(self, adj, num_tokens=2):
        B, N, _ = adj.shape
        new_N = N + num_tokens
        new_adj = torch.zeros(B, new_N, new_N, device=adj.device)
        new_adj[:, num_tokens:, num_tokens:] = adj
        # 两个 token（disease/parc）与所有节点弱连（值=1，可按需改小，如0.5）
        for i in range(num_tokens):
            new_adj[:, i, :] = 1
            new_adj[:, :, i] = 1
        # 零对角
        new_adj = new_adj - torch.diag_embed(torch.diagonal(new_adj, dim1=1, dim2=2))
        return new_adj

    def forward(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None):
        """
        node_features: [B,N,F]
        Adj_block:     [B,N,N] （0/1 或权重）
        parc_type:     str（键需在 self.parcellation_tokens 里）
        disease_type:  str（自动转小写，不存在则用 'none'）
        valid_num_nodes: List[int]（每个样本有效节点数）
        """
        B, N, F = node_features.shape
        device = node_features.device
        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        # 1) RWSE 拼接
        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)                 # [B,N,k]
        node_features = torch.cat([node_features, rwse], dim=-1)               # [B,N,F+k]

        # 2) Zero-pad 到 max_feature_dim
        padded = torch.zeros((B, N, self.max_feature_dim), device=device)      # [B,N,Dm]
        padded[:, :, :node_features.shape[-1]] = node_features

        # 3) 准备 tokens（投影）
        parc_token = self.projection_layer(self.parcellation_tokens[parc_type].expand(B, 1, -1))  # [B,1,H]
        disease_type = disease_type.lower()
        if disease_type not in self.disease_embeddings:
            disease_type = 'none'
        disease_token = self.disease_proj(self.disease_embeddings[disease_type].expand(B, 1, -1)) # [B,1,H]

        # 4) GraphPrompt（在 projection 前应用到特征/边）
        x_prompted, A_tilde, attn_bias_nodes = self.graph_prompt(
            node_feats_BND=padded,               # [B,N,Dm]
            adj_BNN=Adj_block.float(),           # [B,N,N]
            disease_token_B1H=disease_token,     # [B,1,H]
            parc_token_B1H=parc_token,           # [B,1,H]
            valid_num_nodes=valid_num_nodes
        )

        # 5) 投影到 hidden_dim
        x_proj = self.projection_layer(x_prompted)                              # [B,N,H]

        # 6) 拼接 tokens & 邻接扩展
        x = torch.cat([disease_token, parc_token, x_proj], dim=1)              # [B,N+2,H]
        Adj_block_with_tokens = self.expand_adj_block(A_tilde, num_tokens=2)   # [B,N+2,N+2]

        # 7) padding mask
        padding_mask = torch.ones(B, N + 2, device=device).bool()
        for i, n_valid in enumerate(valid_num_nodes):
            padding_mask[i, :n_valid + 2] = False

        # 8) 堆叠 UGFormer + FastMoEGCN
        out = 0
        for i in range(len(self.ugformer_layers)):
            # 需要 batched attn_mask 时，可把 attn_bias_nodes pad 成 [B,L,L] 并传 src_mask=
            h = self.ugformer_layers[i](x, src_key_padding_mask=padding_mask)
            node_h = h[:, 2:, :]
            node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()
            node_h = self.lst_gnn[i](node_h, Adj_block_with_tokens[:, 2:, 2:])
            g = (node_h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            out += self.predictions[i](self.dropouts[i](g))
        return g  # 返回图表示（你外面还有一个分类头）

# ===========================================
# Disease Graph Classifier 封装
# ===========================================
class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj, parc_type, disease_type, valid_num_nodes=None):
        g = self.encoder(x, adj, parc_type, disease_type, valid_num_nodes)
        return self.classifier(g)



# ======= Step 4: Run Example (GraphPrompt-enabled) =======
if __name__ == "__main__":
    import torch

    def build_symmetric_adj(B, N, density=0.25, device="cpu", seed=0):
        """
        生成给定稀疏度的对称邻接（零对角）。
        density 为非对角元素的期望密度（0~1）。
        """
        g = torch.Generator(device=device).manual_seed(seed)
        m = torch.rand(B, N, N, generator=g, device=device)
        m = (m + m.transpose(1, 2)) / 2
        # 只对上三角阈值化以匹配 density
        triu = torch.triu_indices(N, N, offset=1)
        flat = m[:, triu[0], triu[1]]  # [B, N*(N-1)/2]
        kth = torch.quantile(flat, 1 - density, dim=1, keepdim=True)
        keep = (flat >= kth).float()
        adj = torch.zeros(B, N, N, device=device)
        adj[:, triu[0], triu[1]] = keep
        adj = adj + adj.transpose(1, 2)
        # 零对角
        adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))
        return adj

    def offdiag_density(adj):
        B, N, _ = adj.shape
        return ((adj.sum(dim=(1,2))) / (N * (N - 1))).tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(2025)
    print("Using device:", device)

    model = BrainGFM(
        ff_hidden_size=64,
        num_classes=2,
        num_self_att_layers=2,
        dropout=0.3,
        num_GNN_layers=2,
        nhead=4,
        hidden_dim=128,
        max_feature_dim=256,
        rwse_steps=5,
        max_nodes=256,
        moe_num_experts=4
    ).to(device)

    classifier = DiseaseGraphClassifier(model, hidden_dim=128, num_classes=2).to(device)

    # ==== Fake data ====
    B, N, F = 4, 99, 111
    x = torch.randn(B, N, F, device=device)
    adj = build_symmetric_adj(B, N, density=0.25, device=device, seed=7)

    # 每个样本的有效节点数（测试 padding/mask）
    valid_num_nodes = [99, 90, 85, 95]
    # 把无效节点对应的行列置零，模拟真实预处理
    for i, n in enumerate(valid_num_nodes):
        if n < N:
            adj[i, n:, :] = 0.0
            adj[i, :, n:] = 0.0

    print("Input shapes -> x:", tuple(x.shape), "adj:", tuple(adj.shape))
    print("Adj off-diagonal density (before prompt):", [round(d, 4) for d in offdiag_density(adj)])

    # ==== Forward ====
    model.eval()
    with torch.no_grad():
        logits = classifier(x, adj, parc_type='schaefer', disease_type='MDD', valid_num_nodes=valid_num_nodes)

    print("Logits shape:", tuple(logits.shape))
    print("Logits:", logits)