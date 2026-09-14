# braingfm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if self.num_experts == 1:
            return self.experts[0](x)
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
    def __init__(self, in_features, out_features, act=torch.relu, bias=False, layer_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.layer_norm = layer_norm
        self.act = F.gelu if layer_norm else act
        self.bn = nn.LayerNorm(out_features) if layer_norm else nn.BatchNorm1d(out_features)
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
        out = self.bn(out) if self.layer_norm else self.bn(out.view(-1, out.shape[-1])).view(out.shape)
        return self.act(out)


class FastMoEGCN(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, layer_norm=False):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, layer_norm=layer_norm)
            for _ in range(num_experts)
        ])

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, H]
            adj: [B, N, N]
        """
        if self.num_experts == 1:
            return self.experts[0](x, adj)
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
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, num_experts=4, prenorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FastMoEFFN(d_model, dim_feedforward, num_experts, dropout)
        self.prenorm = prenorm

    def forward(self, src, src_mask=None, is_causal=None, src_key_padding_mask=None):
        if self.prenorm:
            h = self.norm1(src)
            attn_out, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask, attn_mask=src_mask, need_weights=False)
            src = src + self.dropout(attn_out)
            src = src + self.dropout(self.ffn(self.norm2(src)))
            return src
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
                 hidden_dim=128, max_feature_dim=256, rwse_steps=5, max_nodes=512, moe_num_experts=4,
                 gcn_residual=False, gcn_norm=False, gcn_layer_norm=False, prenorm=False, attn_bias=False,
                 readout="mean", rwse_fixed=False, token_init=1.0, n_clusters=10, node_id_emb=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_feature_dim = max_feature_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes
        self.nhead = nhead
        self.gcn_residual, self.gcn_norm, self.prenorm, self.attn_bias = gcn_residual, gcn_norm, prenorm, attn_bias
        self.readout, self.rwse_fixed, self.node_id_emb = readout, rwse_fixed, node_id_emb
        self.out_dim = {"meanmax_ln": 2 * hidden_dim, "ocread": n_clusters * hidden_dim}.get(readout, hidden_dim)

        # Project input features to hidden_dim
        self.projection_layer = nn.Linear(self.max_feature_dim, self.hidden_dim)
        self.disease_proj = nn.Linear(768, self.hidden_dim)

        # Parcellation tokens (stored in original feature space; projected later)
        self.parcellation_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, self.max_feature_dim) * token_init)
            for name in ['schaefer', 'schaefer200', 'schaefer300', 'shen268', 'power264', 'gordon333', 'aal116', 'aal3v1']
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
                    dim_feedforward=ff_hidden_size, dropout=dropout, num_experts=moe_num_experts, prenorm=prenorm
                ),
                num_layers=num_self_att_layers
            )
            for _ in range(num_GNN_layers)
        ])

        # Stacked MoE-GCN blocks
        self.lst_gnn = nn.ModuleList([FastMoEGCN(hidden_dim, moe_num_experts, layer_norm=gcn_layer_norm) for _ in range(num_GNN_layers)])
        if node_id_emb:
            parc_sizes = {"schaefer": 100, "schaefer200": 200, "schaefer300": 300, "shen268": 268, "power264": 264,
                          "gordon333": 333, "aal116": 116, "aal3v1": 166}
            self.node_id_embs = nn.ParameterDict({k: nn.Parameter(torch.randn(1, n, hidden_dim) * token_init) for k, n in parc_sizes.items()})
        if attn_bias:
            self.attn_bias_adj = nn.Parameter(torch.zeros(nhead))
            self.attn_bias_prompt = nn.Parameter(torch.zeros(nhead))
        if readout == "meanmax_ln":
            self.readout_ln = nn.LayerNorm(hidden_dim)
        if readout == "ocread":
            self.readout_ln = nn.LayerNorm(hidden_dim)
            self.oc_centres = nn.Parameter(torch.empty(n_clusters, hidden_dim))
            nn.init.orthogonal_(self.oc_centres)
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
        new_adj[:, :num_tokens, :] = 1
        new_adj[:, :, :num_tokens] = 1

        # Remove self-loops (zero diagonal)
        new_adj = new_adj - torch.diag_embed(torch.diagonal(new_adj, dim1=1, dim2=2))
        return new_adj

    def forward(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None, return_nodes=False):
        """
        Args:
            node_features:   [B, N, F]
            Adj_block:       [B, N, N] (binary or weighted adjacency)
            parc_type:       str (must be a key in self.parcellation_tokens)
            disease_type:    str (lower-cased internally; fallback to 'none' if missing)
            valid_num_nodes: List[int] (number of valid nodes per sample; used for padding mask)
            return_nodes:    if True also return the per-node states [B, N, H] used by the readout
                             (needed for node-level pre-training objectives)
        """
        B, N, F = node_features.shape
        device = node_features.device
        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)                 # [B, N, k]

        padded = torch.zeros((B, N, self.max_feature_dim), device=device)      # [B, N, Dm]
        if self.rwse_fixed:
            padded[:, :, :node_features.shape[-1]] = node_features
            padded[:, :, self.max_feature_dim - self.rwse_steps:] = rwse
        else:
            node_features = torch.cat([node_features, rwse], dim=-1)
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

        x_proj = self.projection_layer(x_prompted)                              # [B, N, H]
        if self.node_id_emb:
            x_proj = x_proj + self.node_id_embs[parc_type][:, :N, :]

        # 6) Concatenate tokens & expand adjacency
        x = torch.cat([disease_token, parc_token, x_proj], dim=1)              # [B, N+2, H]
        Adj_block_with_tokens = self.expand_adj_block(A_tilde, num_tokens=2)   # [B, N+2, N+2]

        # 7) Build padding mask
        n_valid_t = torch.as_tensor(valid_num_nodes, device=device).view(B, 1) + 2
        padding_mask = torch.arange(N + 2, device=device).view(1, -1) >= n_valid_t

        node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()
        A_gcn = Adj_block_with_tokens[:, 2:, 2:]
        if self.gcn_norm:
            A_hat = A_gcn + torch.eye(N, device=device).unsqueeze(0)
            dinv = A_hat.sum(-1).clamp(min=1e-6).rsqrt()
            A_gcn = dinv.unsqueeze(-1) * A_hat * dinv.unsqueeze(1)
        attn_mask = None
        if self.attn_bias:
            bias = torch.zeros(B, self.nhead, N + 2, N + 2, device=device)
            bias[:, :, 2:, 2:] = (A_tilde.unsqueeze(1) * self.attn_bias_adj.view(1, -1, 1, 1)
                                  + attn_bias_nodes.unsqueeze(1) * self.attn_bias_prompt.view(1, -1, 1, 1))
            attn_mask = bias.reshape(B * self.nhead, N + 2, N + 2)
        for i in range(len(self.ugformer_layers)):
            for layer in self.ugformer_layers[i].layers:
                x = layer(x, src_mask=attn_mask, src_key_padding_mask=padding_mask)
            node_in = x[:, 2:, :]
            node_h = self.lst_gnn[i](node_in, A_gcn) * node_mask
            if self.gcn_residual:
                node_h = node_in + node_h
            x = torch.cat([x[:, :2, :], node_h], dim=1)
        if self.readout == "ocread":
            z = self.readout_ln(node_h)
            P = torch.softmax(torch.einsum("bnh,kh->bnk", z, self.oc_centres) / math.sqrt(z.shape[-1]), dim=-1) * node_mask
            pooled = torch.einsum("bnk,bnh->bkh", P, z) / (P.sum(1).unsqueeze(-1) + 1e-6)
            g = pooled.flatten(1)
        elif self.readout == "meanmax_ln":
            z = self.readout_ln(node_h)
            mean = (z * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            mx = z.masked_fill(node_mask == 0, float("-inf")).max(dim=1).values
            g = torch.cat([mean, mx], dim=-1)
        else:
            z = node_h
            g = node_h.sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
        if return_nodes:
            return g, z
        return g  # Return graph embedding (external classifier head is used outside)

# ===========================================
# DiseaseGraphClassifier wrapper
# ===========================================
class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        in_dim = getattr(encoder, "out_dim", hidden_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

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
