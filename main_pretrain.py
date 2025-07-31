import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from BrainGFM_autoencoder import GraphMaskedAutoencoder
from BrainGFM import BrainGFM


class ExP():
    def __init__(self, nsub, pretrain_mode="gmae+gcl"):
        super(ExP, self).__init__()
        self.batch_size = 128
        self.n_epochs = 100
        self.lr = 0.0001
        self.b1 = 0.5
        self.b2 = 0.99
        self.nSub = nsub

        self.pretrain_mode = pretrain_mode.lower()
        self.save_path = f'./exp_results/fmri/graph_mae_pretrain/{self.pretrain_mode}/'
        os.makedirs(self.save_path, exist_ok=True)

        self.max_feature_dim = 512
        self.hidden_dim = None  # ✅ 动态设置为 feat_dim
        self.model = None
        self.optimizer = None

    def get_embarc_graph_data(self, path):
        return np.load(path, allow_pickle=True)

    def init_model(self, num_nodes, feat_dim):
        self.hidden_dim = feat_dim  # ✅ 确保 decoder 输出维度 = 输入特征维度

        encoder = BrainGFM(
            ff_hidden_size=256,
            num_classes=2,
            num_self_att_layers=4,
            dropout=0.3,
            num_GNN_layers=4,
            nhead=8,
            hidden_dim=128,  # encoder 输出维度仍保持固定
            max_feature_dim=self.max_feature_dim,
            rwse_steps=5,
            moe_num_experts=1
        )
        self.model = GraphMaskedAutoencoder(
            encoder=encoder,
            hidden_dim=self.hidden_dim,  # 传入 decoder 输出特征维度
            pretrain_mode=self.pretrain_mode
        ).cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

    def train(self, data_t):
        node_feat = torch.from_numpy(data_t).float()
        B, N, F = node_feat.shape
        self.init_model(N, F)

        dataset = TensorDataset(node_feat)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_epoch, loss_epoch_rec, loss_epoch_cl = [], [], []

        for epoch in range(self.n_epochs):
            self.model.train()
            losses, rec_losses, cl_losses = [], [], []
            print(f"\n=== Epoch {epoch + 1}/{self.n_epochs} ===")

            for batch_idx, (node_feat_batch,) in enumerate(dataloader):
                node_feat_batch = node_feat_batch.cuda()
                B, N, F = node_feat_batch.shape
                adj_batch = (torch.rand(B, N, N) > 0.3).float().cuda()  # 可替换为真实 adj

                total_loss, rec_loss, cl_loss, _, _ = self.model(
                    node_feat_batch,
                    adj_batch,
                    parc_type="schaefer",
                    disease_type="MDD",
                    current_epoch=epoch
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                losses.append(total_loss.item())
                rec_losses.append(rec_loss.item())
                cl_losses.append(cl_loss.item())

                print(f"    Batch {batch_idx + 1}/{len(dataloader)} "
                      f"| Total: {total_loss.item():.6f} "
                      f"| Rec: {rec_loss.item():.6f} "
                      f"| CL: {cl_loss.item():.6f}")

            epoch_loss = np.mean(losses)
            rec_epoch_loss = np.mean(rec_losses)
            cl_epoch_loss = np.mean(cl_losses)
            print(f"===> Epoch {epoch + 1} Finished | Avg Loss: {epoch_loss:.6f}  "
                  f"Avg Rec: {rec_epoch_loss:.6f}   Avg CL: {cl_epoch_loss:.6f}")

            torch.save(self.model.state_dict(), os.path.join(self.save_path, f'graphmae_{self.pretrain_mode}.pth'))
            loss_epoch.append(epoch_loss)
            loss_epoch_rec.append(rec_epoch_loss)
            loss_epoch_cl.append(cl_epoch_loss)

        np.save(os.path.join(self.save_path, 'loss_epoch.npy'), np.array(loss_epoch))
        np.save(os.path.join(self.save_path, 'loss_epoch_rec.npy'), np.array(loss_epoch_rec))
        np.save(os.path.join(self.save_path, 'loss_epoch_cl.npy'), np.array(loss_epoch_cl))

        print("\n=== Final Loss per Epoch ===")
        for i, l in enumerate(loss_epoch):
            print(f"Epoch {i + 1}: Loss = {l:.6f}")

        return loss_epoch[-1]


def main():
    path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/atlas_group/test2' # 数据路径
    pretrain_mode = "gmae+gcl"  # 可选：gmae, gcl, gmae+gcl

    exp = ExP(nsub=1, pretrain_mode=pretrain_mode)
    atlas_all = os.listdir(path_t)

    for data_atlas in atlas_all:
        data_path = os.path.join(path_t, data_atlas)
        total_data_t = exp.get_embarc_graph_data(data_path)

        print(f'\n*********** Training on {data_atlas} ***********')
        loss_value = exp.train(total_data_t)

    print(f'\n=== Pre-training with {pretrain_mode.upper()} Done! ===')


if __name__ == "__main__":
    main()