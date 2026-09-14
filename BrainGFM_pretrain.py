import torch, torch.nn as nn, torch.nn.functional as F
from BrainGFM_Gprompt import BrainGFM


def build_encoder(node_id_emb=True):
    return BrainGFM(ff_hidden_size=512, num_classes=2, num_self_att_layers=4, dropout=0.2, num_GNN_layers=4, nhead=8,
                    hidden_dim=256, max_feature_dim=512, rwse_steps=5, moe_num_experts=1,
                    gcn_residual=True, gcn_norm=True, gcn_layer_norm=True, prenorm=True, attn_bias=True,
                    readout="meanmax_ln", rwse_fixed=True, token_init=0.02, node_id_emb=node_id_emb)


class NTXent(nn.Module):
    def __init__(self, t):
        super().__init__(); self.t = t

    def forward(self, a, b, group=None):
        B = a.shape[0]; z = F.normalize(torch.cat([a, b]), dim=1); sim = z @ z.T / self.t
        excl = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        if group is not None:
            g2 = torch.cat([group, group]); diff = g2.unsqueeze(0) != g2.unsqueeze(1)
            pos = torch.zeros_like(excl); ar = torch.arange(B, device=z.device); pos[ar, ar + B] = True; pos[ar + B, ar] = True
            excl = excl | (diff & ~pos)
        sim = sim.masked_fill(excl, -1e4)
        target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, target)


class Pretrainer(nn.Module):
    def __init__(self, encoder, max_feature_dim=512, dec_layers=2, dec_heads=4, dropout=0.1, temperature=0.2):
        super().__init__()
        self.encoder = encoder; H = encoder.hidden_dim
        self.mask_feat = nn.Parameter(torch.zeros(max_feature_dim))
        self.dec_mask = nn.Parameter(torch.zeros(1, 1, H)); nn.init.normal_(self.dec_mask, std=0.02)
        layer = nn.TransformerEncoderLayer(H, dec_heads, dim_feedforward=2 * H, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(layer, dec_layers, enable_nested_tensor=False)
        self.dec_ln = nn.LayerNorm(H); self.proj_out = nn.Linear(H, max_feature_dim)
        self.proj_cl = nn.Sequential(nn.Linear(encoder.out_dim, H), nn.GELU(), nn.Linear(H, 128))
        self.adj_scale = nn.Parameter(torch.tensor(5.0)); self.adj_bias = nn.Parameter(torch.tensor(-2.0))
        self.ntxent = NTXent(temperature)

    @staticmethod
    def thr_adj(x, thr):
        a = (x > thr).to(x.dtype); return (a + a.transpose(1, 2)) / 2

    @staticmethod
    def light_aug(x, adj, p_edge=0.1, noise=0.02):
        keep = (torch.rand_like(adj) > p_edge).to(adj.dtype); keep = torch.maximum(keep, keep.transpose(1, 2))
        return x + noise * torch.randn_like(x), adj * keep

    def mask_view(self, x, adj, ratio, mask_adj=True):
        B, N, _ = x.shape
        n_mask = max(1, int(round(N * ratio)))
        ids = torch.rand(B, N, device=x.device).argsort(1)[:, :n_mask]
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device).scatter_(1, ids, True)
        keep = (~mask).to(x.dtype)
        x_in = x * keep.unsqueeze(1)
        x_in = torch.where(mask.unsqueeze(-1), self.mask_feat[:N].to(x.dtype).view(1, 1, N).expand(B, N, N), x_in)
        adj_in = adj * keep.unsqueeze(1) * keep.unsqueeze(2) if mask_adj else adj
        return x_in, adj_in, mask

    def encode_masked(self, x, ratio, parc, mask_adj, thr, need_nodes=False):
        adj = self.thr_adj(x, thr)
        x_in, adj_in, mask = self.mask_view(x, adj, ratio, mask_adj)
        out = self.encoder(x_in, adj_in, parc, "none", return_nodes=need_nodes)
        return out, adj, mask

    def forward(self, xa, xb, parc, ratio, mask_adj=True, thr=0.3, group=None):
        B, N, _ = xa.shape
        (g1, z), adj_a, mask = self.encode_masked(xa, ratio, parc, mask_adj, thr, need_nodes=True)
        z_dec = torch.where(mask.unsqueeze(-1), self.dec_mask.to(z.dtype).expand(B, N, -1), z)
        pred = self.proj_out(self.dec_ln(self.decoder(z_dec)))[..., :N]
        rec = F.smooth_l1_loss(pred[mask].float(), xa[mask].float())
        zn = F.normalize(z.float(), dim=-1)
        logits = torch.einsum("bnh,bmh->bnm", zn, zn) * self.adj_scale + self.adj_bias
        rowmask = mask.unsqueeze(-1) | mask.unsqueeze(1)
        adj_loss = F.binary_cross_entropy_with_logits(logits[rowmask], adj_a.float()[rowmask])
        adj_b = self.thr_adj(xb, thr)
        xb_in, adjb_in, _ = self.mask_view(xb, adj_b, ratio, mask_adj)
        xb_in, adjb_in = self.light_aug(xb_in, adjb_in)
        g2 = self.encoder(xb_in, adjb_in, parc, "none")
        cl = self.ntxent(self.proj_cl(g1).float(), self.proj_cl(g2).float(), group)
        return rec, adj_loss, cl, g1

    def xatlas(self, g1_sub, xc, parc_c, thr, group=None):
        adj_c = self.thr_adj(xc, thr); xc, adj_c = self.light_aug(xc, adj_c)
        g3 = self.encoder(xc, adj_c, parc_c, "none")
        return self.ntxent(self.proj_cl(g1_sub).float(), self.proj_cl(g3).float(), group)
