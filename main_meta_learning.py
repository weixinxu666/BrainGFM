import torch
import torch.nn as nn
import numpy as np
import random
import higher
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from BrainGFM import BrainGFM


def build_adj(x, threshold=0.6):
    return (x > threshold).float()


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else 0.0
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    return acc, auc, f1, precision, recall, sensitivity, specificity


def normalize_conn(x):
    return (x - x.mean(axis=(-1, -2), keepdims=True)) / (x.std(axis=(-1, -2), keepdims=True) + 1e-6)


class MAMLMetaLearnerANIL:
    def __init__(self, data_path, data_train_names, data_test_names):
        self.data_all = np.load(data_path, allow_pickle=True).item()
        self.data_train_names = data_train_names
        self.data_test_names = data_test_names

        self.task_to_disease = {
            'adni2': 'AD', 'abide1': 'ASD', 'abide2': 'ASD', 'adhd200': 'ADHD',
            'adni2_aal116': 'AD', 'hbn_mdd': 'MDD', 'hbn_ptsd': 'PTSD',
            'ucla_schz': 'SCHZ', 'ucla_bp': 'BP'
        }
        self.parc_type = 'schaefer'

        # Backbone encoder
        self.encoder = BrainGFM(
            ff_hidden_size=64, num_classes=2, num_self_att_layers=2, dropout=0.3,
            num_GNN_layers=2, nhead=4, hidden_dim=128, max_feature_dim=256,
            rwse_steps=5, max_nodes=256, moe_num_experts=4
        ).cuda()

        # Freeze parcellation tokens
        for name, param in self.encoder.named_parameters():
            if 'parcellation_tokens' in name:
                param.requires_grad = False

        self.meta_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=5e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def meta_train(self, meta_epochs=20, meta_batch_size=4, inner_steps=5, inner_lr=1e-3):
        self.encoder.train()
        for epoch in range(meta_epochs):
            tasks = random.sample(self.data_train_names, meta_batch_size)
            meta_loss_all = 0.0

            for task in tasks:
                disease = self.task_to_disease[task].lower()
                data = self.data_all[task]
                X, y = normalize_conn(data['conn']), data['label']

                if min(np.bincount(y)) < 6:
                    continue

                spt_idx, qry_idx = [], []
                for c in np.unique(y):
                    c_idx = np.where(y == c)[0]
                    np.random.shuffle(c_idx)
                    spt_idx += list(c_idx[:5])
                    qry_idx += list(c_idx[5:])

                x_spt, y_spt = torch.FloatTensor(X[spt_idx]).cuda(), torch.LongTensor(y[spt_idx]).cuda()
                x_qry, y_qry = torch.FloatTensor(X[qry_idx]).cuda(), torch.LongTensor(y[qry_idx]).cuda()
                adj_spt, adj_qry = build_adj(x_spt).cuda(), build_adj(x_qry).cuda()
                nodes_spt = [x_spt.shape[1]] * x_spt.shape[0]
                nodes_qry = [x_qry.shape[1]] * x_qry.shape[0]

                with torch.no_grad():
                    feat_spt = self.encoder(x_spt, adj_spt, self.parc_type, disease, nodes_spt)
                    feat_qry = self.encoder(x_qry, adj_qry, self.parc_type, disease, nodes_qry)

                classifier = nn.Linear(feat_spt.shape[-1], 2).cuda()
                inner_optim = torch.optim.SGD(classifier.parameters(), lr=inner_lr)

                with higher.innerloop_ctx(classifier, inner_optim, copy_initial_weights=True) as (fmodel, diffopt):
                    for _ in range(inner_steps):
                        pred = fmodel(feat_spt)
                        loss = self.loss_fn(pred, y_spt)
                        diffopt.step(loss)

                    pred_q = fmodel(feat_qry)
                    loss_q = self.loss_fn(pred_q, y_qry)
                    loss_q.backward()
                    acc = (pred_q.argmax(1) == y_qry).float().mean().item()
                    meta_loss_all += loss_q.item()
                    print(f"[Epoch {epoch}] Task {task} | Loss: {loss_q.item():.4f} | Acc: {acc:.4f}")

                self.meta_optimizer.step()

            print(f"==> Meta Epoch {epoch} Avg Loss: {meta_loss_all / meta_batch_size:.4f}")

    def meta_test(self, shots=5):
        self.encoder.eval()
        print("\n[Meta Test]")
        for task in self.data_test_names:
            print(f"→ Testing on {task}")
            disease = self.task_to_disease[task].lower()
            data = self.data_all[task]
            X, y = normalize_conn(data['conn']), data['label']
            if min(np.bincount(y)) <= shots:
                print("  Skipped: insufficient class samples.")
                continue

            spt_idx, qry_idx = [], []
            for c in np.unique(y):
                c_idx = np.where(y == c)[0]
                np.random.shuffle(c_idx)
                spt_idx += list(c_idx[:shots])
                qry_idx += list(c_idx[shots:])

            x_spt, y_spt = torch.FloatTensor(X[spt_idx]).cuda(), torch.LongTensor(y[spt_idx]).cuda()
            x_qry, y_qry = torch.FloatTensor(X[qry_idx]).cuda(), torch.LongTensor(y[qry_idx]).cuda()
            adj_spt, adj_qry = build_adj(x_spt).cuda(), build_adj(x_qry).cuda()
            nodes_spt = [x_spt.shape[1]] * x_spt.shape[0]
            nodes_qry = [x_qry.shape[1]] * x_qry.shape[0]

            with torch.no_grad():
                feat_spt = self.encoder(x_spt, adj_spt, self.parc_type, disease, nodes_spt)
                feat_qry = self.encoder(x_qry, adj_qry, self.parc_type, disease, nodes_qry)

            classifier = nn.Linear(feat_spt.shape[-1], 2).cuda()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
            for _ in range(10):
                loss = self.loss_fn(classifier(feat_spt), y_spt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                y_pred = classifier(feat_qry).argmax(1).cpu().numpy()
                y_true = y_qry.cpu().numpy()
                acc, auc, f1, *_ = compute_metrics(y_true, y_pred)
                print(f"   Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")


if __name__ == '__main__':
    data_path = '/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data_maml/maml_all.npy'
    train_tasks = ['adni2', 'abide1', 'abide2', 'adhd200', 'hbn_mdd']
    test_tasks = ['adni2_aal116', 'ucla_schz', 'ucla_bp']

    learner = MAMLMetaLearnerANIL(data_path, train_tasks, test_tasks)
    learner.meta_train(meta_epochs=30)
    print("=" * 80)
    learner.meta_test(shots=50)