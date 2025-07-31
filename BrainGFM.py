import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder
from transformers import AutoTokenizer, AutoModel
from disease_names import disease_names

# ======= Step 1: 获取疾病 token from ClinicalBERT =======
def get_disease_embeddings():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for disease, name in disease_names.items():
            inputs = tokenizer(name, return_tensors='pt')
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings[disease] = cls_embedding.squeeze(0)

    return embeddings

# ======= Step 2: FastMoE 组件 =======
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
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))
        top1 = torch.argmax(scores, dim=-1)
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
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.size(0), -1, -1))
        out = torch.bmm(adj, support)
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
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))
        top1 = torch.argmax(scores, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx], adj[idx])
        return out

class GTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FastMoEFFN(d_model, dim_feedforward, num_experts, dropout)

    def forward(self, src, src_mask=None, is_causal=None, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src

# ======= Step 3: Main Model with Attention Mask =======
class BrainGFM(nn.Module):
    def __init__(self, ff_hidden_size, num_classes, num_self_att_layers, dropout, num_GNN_layers, nhead,
                 hidden_dim=128, max_feature_dim=256, rwse_steps=5, max_nodes=256, moe_num_experts=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_feature_dim = max_feature_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes

        self.projection_layer = nn.Linear(self.max_feature_dim, self.hidden_dim)
        self.disease_proj = nn.Linear(768, self.hidden_dim)

        self.parcellation_tokens = nn.ParameterDict({
            'schaefer': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer200': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer300': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'shen268': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'power264': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'gordon333': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal116': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal3v1': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
        })

        self.node_prompt = nn.Parameter(torch.randn(1, self.max_nodes, self.max_feature_dim))
        disease_embed_dict = get_disease_embeddings()
        # ✅ 替代方式：改为普通 dict，并手动注册每个 Parameter
        self.disease_embeddings = {}
        for k, v in disease_embed_dict.items():
            param = nn.Parameter(v.unsqueeze(0).unsqueeze(0))  # shape [1, 1, 768]
            self.disease_embeddings[k.lower()] = param  # 统一为小写
            self.register_parameter(f'disease_embedding_{k.lower()}', param)

        self.ugformer_layers = nn.ModuleList([
            TransformerEncoder(GTransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead,
                dim_feedforward=ff_hidden_size, dropout=dropout, num_experts=moe_num_experts),
                num_layers=num_self_att_layers)
            for _ in range(num_GNN_layers)
        ])
        self.lst_gnn = nn.ModuleList([FastMoEGCN(hidden_dim, moe_num_experts) for _ in range(num_GNN_layers)])
        self.predictions = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_GNN_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_GNN_layers)])

    def compute_rwse(self, adj, k):
        B, N, _ = adj.shape
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw = adj.clone()
        diag_features = []
        for _ in range(k):
            rw_diag = torch.diagonal(rw, dim1=1, dim2=2).unsqueeze(-1)
            diag_features.append(rw_diag)
            rw = torch.bmm(rw, adj)
        return torch.cat(diag_features, dim=-1)

    def expand_adj_block(self, adj, num_tokens=2):
        B, N, _ = adj.shape
        new_N = N + num_tokens
        new_adj = torch.zeros(B, new_N, new_N, device=adj.device)
        new_adj[:, num_tokens:, num_tokens:] = adj
        for i in range(num_tokens):
            new_adj[:, i, :] = 1
            new_adj[:, :, i] = 1
        return new_adj

    def forward(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None):
        B, N, F = node_features.shape
        device = node_features.device

        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)
        node_features = torch.cat([node_features, rwse], dim=-1)

        padded = torch.zeros((B, N, self.max_feature_dim), device=device)
        padded[:, :, :node_features.shape[-1]] = node_features

        B, N_aug, D = padded.shape
        prompt = self.node_prompt
        if prompt.size(1) < N_aug:
            repeat_prompt = prompt[:, -1:, :].expand(1, N_aug - prompt.size(1), D)
            prompt = torch.cat([prompt, repeat_prompt], dim=1)
        prompt = prompt[:, :N_aug, :].expand(B, -1, -1)
        padded = padded * prompt

        x_proj = self.projection_layer(padded)
        parc_token = self.projection_layer(self.parcellation_tokens[parc_type].expand(B, 1, -1))
        disease_type = disease_type.lower()
        if disease_type not in self.disease_embeddings:
            disease_type = 'none'

        disease_token = self.disease_proj(self.disease_embeddings[disease_type].expand(B, 1, -1))

        x = torch.cat([disease_token, parc_token, x_proj], dim=1)
        Adj_block = self.expand_adj_block(Adj_block, num_tokens=2)

        padding_mask = torch.ones(B, N + 2, device=device).bool()
        for i, n_valid in enumerate(valid_num_nodes):
            padding_mask[i, :n_valid + 2] = False

        out = 0
        for i in range(len(self.ugformer_layers)):
            h = self.ugformer_layers[i](x, src_key_padding_mask=padding_mask)
            node_h = h[:, 2:, :]
            node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()  # shape [B, N, 1]
            node_h = self.lst_gnn[i](node_h, Adj_block[:, 2:, 2:])
            g = (node_h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            out += self.predictions[i](self.dropouts[i](g))
        return g

# ======= Disease Graph Classifier Wrapper =======
class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj, parc_type, disease_type, valid_num_nodes=None):
        g = self.encoder(x, adj, parc_type, disease_type, valid_num_nodes)
        return self.classifier(g)

# ======= Step 4: Run Example =======
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    B, N, F = 4, 99, 111
    x = torch.randn(B, N, F).to(device)
    adj = torch.rand(B, N, N).to(device)
    adj = (adj + adj.transpose(1, 2)) / 2
    adj[adj < 0.5] = 0
    adj[adj >= 0.5] = 1

    valid_num_nodes = [99, 90, 85, 95]

    with torch.no_grad():
        logits = classifier(x, adj, parc_type='schaefer', disease_type='MDD', valid_num_nodes=valid_num_nodes)
    print("Logits:", logits)
