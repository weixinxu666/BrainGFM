# braingfm.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from disease_names import disease_names

# Import GraphPrompt module
from graph_prompt import GraphPrompt

# ===========================================
# Disease embeddings extracted by ClinicalBERT (with fallback)
# ===========================================
def get_disease_embeddings(embed_dim=768, seed=42):
    """
    Try to extract CLS embeddings of disease names using ClinicalBERT.
    If failed (e.g., no internet / missing dependency), fall back to random initialization.

    Returns:
        dict[str, torch.Tensor]: mapping disease name -> embedding of shape [768]
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
        # Fallback: deterministic random initialization
        g = torch.Generator().manual_seed(seed)
        embeddings = {}
        for disease in disease_names.keys():
            embeddings[disease.lower()] = torch.randn(embed_dim, generator=g)
        return embeddings


# ===========================================
# FastMoE modules: FFN and GCN
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
        """
        Args:
            x: Tensor of shape [B, N, H]
        """
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B, E]
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
        """
        Args:
            x:   [B, N, Fin]
            adj: [B, N, N]
        """
        support = torch.bmm(
            x,
            self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        )  # [B, N, Fout]
        out = torch.bmm(adj, support)  # [B, N, Fout]
        if self.bias is not None:
            out = out + self.bias
        out = self.bn(out.view(-1, out.shape[-1])).view(out.shape)
        return self.act(out)


class FastMoEGCN(nn.Module):
    def __init__(self, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, H]
            adj: [B, N, N]
        """
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B, E]
        top1 = torch.argmax(scores, dim=-1)  # [B]
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx], adj[idx])
        return out


# ===========================================
# UGFormer encoder layer
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
# BrainGFM encoder (with GraphPrompt)
# ===========================================
class BrainGFM(nn.Module):
    def __init__(self, ff_hidden_size, num_classes, num_self_att_layers, dropout, num_GNN_layers, nhead,
                 hidden_dim=128, max_feature_dim=256, rwse_steps=5, max_nodes=512, moe_num_experts=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_feature_dim = max_feature_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes

        # Project input features to hidden_dim
        self.projection_layer = nn.Linear(self.max_feature_dim, self.hidden_dim)
        self.disease_proj = nn.Linear(768, self.hidden_dim)

        # Parcellation tokens (stored in original feature space; projected later)
        self.parcellation_tokens = nn.ParameterDict({
            'schaefer':    nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer200': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer300': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'shen268':     nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'power264':    nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'gordon333':   nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal116':      nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal3v1':      nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
        })

        # Disease tokens (registered as nn.Parameter)
        disease_embed_dict = get_disease_embeddings()
        self.disease_embeddings = {}
        for k, v in disease_embed_dict.items():
            param = nn.Parameter(v.unsqueeze(0).unsqueeze(0))  # [1,1,768]
            self.disease_embeddings[k.lower()] = param
            self.register_parameter(f'disease_embedding_{k.lower()}', param)

        # Unified GraphPrompt module
        self.graph_prompt = GraphPrompt(
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            max_feature_dim=max_feature_dim,
            node_mode="scale",
            edge_strength=0.3,
            init_std=1e-2
        )

        # Stacked UGFormer blocks
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

        # Stacked MoE-GCN blocks
        self.lst_gnn = nn.ModuleList([FastMoEGCN(hidden_dim, moe_num_experts) for _ in range(num_GNN_layers)])
        self.predictions = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_GNN_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_GNN_layers)])

    def compute_rwse(self, adj, k):
        """
        A simple RWSE implementation using diagonal entries of k-step random walks.
        """
        B, N, _ = adj.shape
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw = adj.clone()
        diag_features = []
        for _ in range(k):
            rw_diag = torch.diagonal(rw, dim1=1, dim2=2).unsqueeze(-1)  # [B, N, 1]
            diag_features.append(rw_diag)
            rw = torch.bmm(rw, adj)
        return torch.cat(diag_features, dim=-1)  # [B, N, k]

    def expand_adj_block(self, adj, num_tokens=2):
        """
        Expand adjacency to include token nodes (disease + parcellation).
        """
        B, N, _ = adj.shape
        new_N = N + num_tokens
        new_adj = torch.zeros(B, new_N, new_N, device=adj.device)
        new_adj[:, num_tokens:, num_tokens:] = adj

        # Weakly connect tokens to all nodes (value=1 by default; can be reduced to e.g. 0.5 if needed)
        for i in range(num_tokens):
            new_adj[:, i, :] = 1
            new_adj[:, :, i] = 1

        # Remove self-loops (zero diagonal)
        new_adj = new_adj - torch.diag_embed(torch.diagonal(new_adj, dim1=1, dim2=2))
        return new_adj

    def forward(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None):
        """
        Args:
            node_features:   [B, N, F]
            Adj_block:       [B, N, N] (binary or weighted adjacency)
            parc_type:       str (must be a key in self.parcellation_tokens)
            disease_type:    str (lower-cased internally; fallback to 'none' if missing)
            valid_num_nodes: List[int] (number of valid nodes per sample; used for padding mask)
        """
        B, N, F = node_features.shape
        device = node_features.device
        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        # 1) Append RWSE features
        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)                 # [B, N, k]
        node_features = torch.cat([node_features, rwse], dim=-1)               # [B, N, F+k]

        # 2) Zero-pad to max_feature_dim
        padded = torch.zeros((B, N, self.max_feature_dim), device=device)      # [B, N, Dm]
        padded[:, :, :node_features.shape[-1]] = node_features

        # 3) Prepare tokens (projected)
        parc_token = self.projection_layer(self.parcellation_tokens[parc_type].expand(B, 1, -1))  # [B,1,H]
        disease_type = disease_type.lower()
        if disease_type not in self.disease_embeddings:
            disease_type = 'none'
        disease_token = self.disease_proj(self.disease_embeddings[disease_type].expand(B, 1, -1)) # [B,1,H]

        # 4) Apply GraphPrompt (before projecting node features to hidden_dim)
        x_prompted, A_tilde, attn_bias_nodes = self.graph_prompt(
            node_feats_BND=padded,               # [B, N, Dm]
            adj_BNN=Adj_block.float(),           # [B, N, N]
            disease_token_B1H=disease_token,     # [B, 1, H]
            parc_token_B1H=parc_token,           # [B, 1, H]
            valid_num_nodes=valid_num_nodes
        )

        # 5) Project to hidden_dim
        x_proj = self.projection_layer(x_prompted)                              # [B, N, H]

        # 6) Concatenate tokens & expand adjacency
        x = torch.cat([disease_token, parc_token, x_proj], dim=1)              # [B, N+2, H]
        Adj_block_with_tokens = self.expand_adj_block(A_tilde, num_tokens=2)   # [B, N+2, N+2]

        # 7) Build padding mask
        padding_mask = torch.ones(B, N + 2, device=device).bool()
        for i, n_valid in enumerate(valid_num_nodes):
            padding_mask[i, :n_valid + 2] = False

        # 8) Stacked UGFormer + FastMoEGCN
        out = 0
        for i in range(len(self.ugformer_layers)):
            # If you need a batched attn_mask, pad attn_bias_nodes to [B, L, L] and pass via src_mask=
            h = self.ugformer_layers[i](x, src_key_padding_mask=padding_mask)
            node_h = h[:, 2:, :]
            node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()
            node_h = self.lst_gnn[i](node_h, Adj_block_with_tokens[:, 2:, 2:])
            g = (node_h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            out += self.predictions[i](self.dropouts[i](g))
        return g  # Return graph embedding (external classifier head is used outside)

# ===========================================
# DiseaseGraphClassifier wrapper
# ===========================================
class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj, parc_type, disease_type, valid_num_nodes=None):
        g = self.encoder(x, adj, parc_type, disease_type, valid_num_nodes)
        return self.classifier(g)


if __name__ == "__main__":
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(2025)
    print("Using device:", device)

    # ===== Build model =====
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

    # ===== Create dummy inputs =====
    B, N, F = 32, 100, 100
    x = torch.randn(B, N, F, device=device)

    # Build symmetric binary adjacency
    adj = torch.rand(B, N, N, device=device)
    adj = (adj + adj.transpose(1, 2)) / 2
    adj[adj < 0.5] = 0
    adj[adj >= 0.5] = 1

    # Remove diagonal (optional)
    adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))

    # Valid node counts (for padding mask); set all to N here to avoid mismatch
    valid_num_nodes = [N] * B

    print("Input shapes -> x:", tuple(x.shape), "adj:", tuple(adj.shape))

    # ===== Count parameters =====
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"[Params] total={total_params:,}  trainable={trainable_params:,}")

    # ===== Estimate FLOPs =====
    flops_counted = False
    try:
        from thop import profile, clever_format
        flops, thop_params = profile(
            classifier,
            inputs=(x, adj, 'schaefer', 'MDD', valid_num_nodes),
            verbose=False
        )
        flops_str, params_str = clever_format([flops, thop_params], "%.3f")
        print(f"[THOP] FLOPs={flops_str}, Params={params_str}")
        flops_counted = True
    except Exception as e:
        print(f"[THOP] Failed to compute: {repr(e)}")

    if not flops_counted:
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count

            flops = FlopCountAnalysis(
                classifier, (x, adj, 'schaefer', 'MDD', valid_num_nodes)
            ).total()
            params_tbl = parameter_count(classifier)

            def humanize(num: float):
                units = ["", "K", "M", "G", "T", "P"]
                idx = 0
                n = float(num)
                while n >= 1000 and idx < len(units) - 1:
                    n /= 1000.0
                    idx += 1
                return f"{n:.3f}{units[idx]}"

            approx_params = params_tbl.get('', total_params) if isinstance(params_tbl, dict) else total_params
            print(f"[fvcore] FLOPs≈{humanize(flops)}  Params≈{humanize(approx_params)}")
            flops_counted = True
        except Exception as e:
            print(f"[fvcore] Failed to compute: {repr(e)}")

    if not flops_counted:
        print("[Info] FLOPs cannot be computed (unsupported ops/dependencies), but params are printed above.")

    # ===== Single forward latency =====
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        logits = classifier(
            x, adj,
            parc_type='schaefer',
            disease_type='MDD',
            valid_num_nodes=valid_num_nodes
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"[Run] Single forward time: {elapsed_ms:.3f} ms  ({elapsed_ms / B:.3f} ms per graph)")

    # ===== Repeated benchmark (warmup + multiple runs) =====
    n_repeats = 50
    warmup = 10
    times_ms = []
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = classifier(
                x, adj,
                parc_type='schaefer',
                disease_type='MDD',
                valid_num_nodes=valid_num_nodes
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            _ = classifier(
                x, adj,
                parc_type='schaefer',
                disease_type='MDD',
                valid_num_nodes=valid_num_nodes
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)
    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0
    p50_ms = float(np.percentile(times_ms, 50))
    p95_ms = float(np.percentile(times_ms, 95))
    per_graph_ms = mean_ms / B
    throughput = 1000.0 * B / mean_ms if mean_ms > 0 else float('inf')

    print(f"[Benchmark] {n_repeats} runs (warmup={warmup})")
    print(f"  • Latency per batch: mean={mean_ms:.3f} ± {std_ms:.3f} ms (p50={p50_ms:.3f}, p95={p95_ms:.3f})")
    print(f"  • Latency per graph: {per_graph_ms:.3f} ms")
    print(f"  • Throughput: {throughput:.1f} graphs/sec")

    print("Logits shape:", tuple(logits.shape))
    print("Done.")
