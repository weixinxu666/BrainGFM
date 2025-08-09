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
        self.batch_size = 64
        self.n_epochs = 20
        self.lr = 0.00005
        self.b1 = 0.5
        self.b2 = 0.99
        self.nSub = nsub

        self.pretrain_sequence = None
        if '+' not in pretrain_mode and '->' in pretrain_mode:
            self.pretrain_sequence = pretrain_mode.split('->')
            self.pretrain_mode = self.pretrain_sequence[0]
        else:
            self.pretrain_mode = pretrain_mode.lower()

        self.save_path = f'./exp_results/fmri/graph_mae_pretrain/{pretrain_mode}/'
        os.makedirs(self.save_path, exist_ok=True)

        self.max_feature_dim = 512
        self.hidden_dim = None
        self.model = None
        self.optimizer = None

    def get_embarc_graph_data(self, path):
        return np.load(path, allow_pickle=True)

    def init_model(self, num_nodes, feat_dim):
        self.hidden_dim = feat_dim

        encoder = BrainGFM(
            ff_hidden_size=256,
            num_classes=2,
            num_self_att_layers=4,
            dropout=0.3,
            num_GNN_layers=4,
            nhead=8,
            hidden_dim=256,
            max_feature_dim=self.max_feature_dim,
            rwse_steps=5,
            moe_num_experts=1
        )

        self.model = GraphMaskedAutoencoder(
            encoder=encoder,
            hidden_dim=self.hidden_dim,
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

        if self.pretrain_sequence:
            total_loss_record = []
            for phase in self.pretrain_sequence:
                print(f"\n=========== Starting {phase.upper()} Phase ===========")
                self.pretrain_mode = phase
                if self.model is None:
                    self.init_model(N, F)
                self.model.set_mode(phase)
                loss_final = self._train_once(node_feat, phase)
                total_loss_record.append(loss_final)
            return total_loss_record[-1]
        else:
            self.init_model(N, F)
            return self._train_once(node_feat, self.pretrain_mode)

    def _train_once(self, node_feat, phase_name):
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

                adj_batch = (node_feat_batch > 0.3).float()
                adj_batch = (adj_batch + adj_batch.transpose(1, 2)) / 2

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

                losses.append(float(total_loss))
                rec_losses.append(float(rec_loss))
                cl_losses.append(float(cl_loss))

                print(f"    Batch {batch_idx + 1}/{len(dataloader)} "
                      f"| Total: {float(total_loss):.6f} "
                      f"| Rec: {float(rec_loss):.6f} "
                      f"| CL: {float(cl_loss):.6f}")

            epoch_loss = np.mean(losses)
            rec_epoch_loss = np.mean(rec_losses)
            cl_epoch_loss = np.mean(cl_losses)
            print(f"===> Epoch {epoch + 1} Finished | Avg Loss: {epoch_loss:.6f}  "
                  f"Avg Rec: {rec_epoch_loss:.6f}   Avg CL: {cl_epoch_loss:.6f}")

            model_filename = f'graphmae_{phase_name}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.save_path, model_filename))
            loss_epoch.append(epoch_loss)
            loss_epoch_rec.append(rec_epoch_loss)
            loss_epoch_cl.append(cl_epoch_loss)

        np.save(os.path.join(self.save_path, f'loss_epoch_{phase_name}.npy'), np.array(loss_epoch))
        np.save(os.path.join(self.save_path, f'loss_epoch_rec_{phase_name}.npy'), np.array(loss_epoch_rec))
        np.save(os.path.join(self.save_path, f'loss_epoch_cl_{phase_name}.npy'), np.array(loss_epoch_cl))

        print(f"\n=== Final Loss for {phase_name.upper()} ===")
        for i, l in enumerate(loss_epoch):
            print(f"Epoch {i + 1}: Loss = {l:.6f}")

        return loss_epoch[-1]


def main():
    path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/atlas_group/test2'

    # 可选模式: "gmae", "gcl", "gmae+gcl", "gmae->gcl", "gcl->gmae"
    pretrain_mode = "gmae->gcl"

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
