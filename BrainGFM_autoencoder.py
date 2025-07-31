import torch
import torch.nn as nn
import torch.nn.functional as F
from BrainGFM import BrainGFM


# === NT-Xent 对比损失 ===
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
        sim.masked_fill_(mask, -9e15)
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
        nom = torch.exp(pos)
        denom = torch.exp(sim).sum(dim=1)
        return -torch.log(nom / denom).mean()


# === GraphMaskedAutoencoder 主体 ===
class GraphMaskedAutoencoder(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, pretrain_mode="gmae+gcl"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.mask_ratio = 0.1  # warm-up 初值
        self.pretrain_mode = pretrain_mode.lower()

        self.decoder_input_dim = encoder.hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_input_dim))
        nn.init.xavier_uniform_(self.mask_token)

        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, self.decoder_input_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_input_dim, hidden_dim)
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.contrastive_loss = NTXentLoss(temperature=0.2)

    def random_mask(self, x, mask_ratio):
        B, N, _ = x.size()
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x_masked, ids_keep, ids_mask, ids_restore

    def view_augmentation(self, node_feat, adj):
        node_feat_aug = node_feat.clone()
        drop_mask = (torch.rand_like(node_feat_aug[..., 0]) < 0.1).unsqueeze(-1)
        node_feat_aug[drop_mask.expand_as(node_feat_aug)] = 0
        edge_mask = (torch.rand_like(adj) > 0.1).float()
        adj_aug = adj * edge_mask
        return node_feat_aug, adj_aug

    def forward(self, node_feat, adj, parc_type, disease_type, current_epoch=0):
        B, N, feat_dim = node_feat.shape
        device = node_feat.device

        # === 动态 warm-up mask ratio ===
        max_ratio = 0.3
        self.mask_ratio = min(max_ratio, 0.05 + 0.025 * current_epoch)

        # === Masked 输入 for GMAE ===
        node_feat_masked, ids_keep, ids_mask, ids_restore = self.random_mask(node_feat, self.mask_ratio)
        adj_masked = torch.gather(adj, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, N))
        adj_masked = torch.gather(adj_masked, dim=2, index=ids_keep.unsqueeze(1).expand(-1, len(ids_keep[0]), -1))
        z_q = self.encoder(node_feat_masked, adj_masked, parc_type, disease_type)
        if z_q.dim() == 2:
            z_q = z_q.unsqueeze(1)

        # === View augmentation for GCL ===
        node_feat_view, adj_view = self.view_augmentation(node_feat, adj)
        z_k = self.encoder(node_feat_view, adj_view, parc_type, disease_type)
        if z_k.dim() == 2:
            z_k = z_k.unsqueeze(1)

        # === 解码重建 for GMAE ===
        mask_len = N - z_q.shape[1]
        mask_tokens = self.mask_token.expand(B, mask_len, z_q.shape[-1])
        x_merged = torch.cat([z_q, mask_tokens], dim=1)
        index_all = torch.cat([ids_keep, ids_mask], dim=1)
        index_all_sorted = torch.argsort(index_all, dim=1)
        x_restored = torch.gather(x_merged, dim=1, index=index_all_sorted.unsqueeze(-1).expand(-1, -1, z_q.shape[-1]))
        pred_feat = self.decoder(x_restored)

        # === Loss ===
        rec_loss, cl_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        if self.pretrain_mode in ["gmae", "gmae+gcl"]:
            for b in range(B):
                rec_loss += self.loss_fn(pred_feat[b, ids_mask[b]], node_feat[b, ids_mask[b]])
            rec_loss /= B

        if self.pretrain_mode in ["gcl", "gmae+gcl"]:
            z_q_flat = F.normalize(z_q, dim=-1).reshape(B * z_q.shape[1], -1)
            z_k_flat = F.normalize(z_k, dim=-1).reshape(B * z_k.shape[1], -1)
            cl_loss = self.contrastive_loss(z_q_flat, z_k_flat)

        if self.pretrain_mode == "gmae":
            total_loss = rec_loss
        elif self.pretrain_mode == "gcl":
            total_loss = 0.01 * cl_loss
        else:
            total_loss = rec_loss + 0.01 * cl_loss

        return total_loss, rec_loss, cl_loss, pred_feat, node_feat


if __name__ == "__main__":
    import torch.optim as optim

    B, N, feat_dim = 8, 77, 123
    x = torch.rand(B, N, feat_dim).cuda()
    adj = (torch.rand(B, N, N) > 0.6).float().cuda()

    encoder = BrainGFM(
        ff_hidden_size=256,
        num_classes=2,
        num_self_att_layers=4,
        dropout=0.3,
        num_GNN_layers=4,
        nhead=8,
        hidden_dim=128,
        max_feature_dim=256,
        rwse_steps=5,
        moe_num_experts=1
    ).cuda()

    for mode in ["gmae", "gcl", "gmae+gcl"]:
        print(f"\n🌟 Training mode: {mode}")
        model = GraphMaskedAutoencoder(encoder, hidden_dim=feat_dim, pretrain_mode=mode).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            total_loss, rec_loss, cl_loss, pred, true = model(
                x, adj, "schaefer", "MDD", current_epoch=epoch
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            print(f"Epoch {epoch:02d} | Total: {total_loss.item():.4f} | Rec: {rec_loss.item():.4f} | CL: {cl_loss.item():.4f} | Mask: {model.mask_ratio:.2f}")