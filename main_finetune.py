import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.backends import cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

from BrainGFM import BrainGFM, DiseaseGraphClassifier

# 设置CUDA
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

cudnn.benchmark = False
cudnn.deterministic = True


# ===============================
# ======== 实验类定义 ============
# ===============================
class ExP:
    def __init__(self, fold_idx, pretrained_path=None):
        super(ExP, self).__init__()

        self.batch_size = 64
        self.n_epochs = 50
        self.lr = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.save_path = './exp_results/fmri/'
        os.makedirs(self.save_path, exist_ok=True)

        self.fold_idx = fold_idx

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = nn.CrossEntropyLoss().cuda()

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

        self.model_t = DiseaseGraphClassifier(
            encoder=encoder,
            hidden_dim=128,
            num_classes=2
        ).cuda()

        # 加载预训练模型（如有）
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            print(f">>> Loading pretrained model from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cuda')
            self.model_t.load_state_dict(state_dict, strict=False)
        elif pretrained_path:
            print(f">>> WARNING: Pretrained model not found at {pretrained_path}")

    def get_data(self, path):
        data = np.load(path, allow_pickle=True).item()
        data_sub = data['adni2']
        return data_sub["conn"], data_sub["label"]

    def train(self, data_t):
        train_data, test_data, train_label, test_label = data_t
        train_label = torch.tensor(train_label, dtype=torch.int64)
        test_label = torch.tensor(test_label, dtype=torch.int64)

        node_feat_train = torch.from_numpy(train_data)
        node_feat_test = torch.from_numpy(test_data)

        adj_train = (node_feat_train > 0.6).int()
        adj_test = (node_feat_test > 0.6).int()

        dataset = TensorDataset(node_feat_train, adj_train, train_label)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model_t.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        node_feat_test = Variable(node_feat_test.type(self.Tensor))
        adj_test = Variable(adj_test.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        best_acc = best_auc = best_score = 0
        y_true, y_pred = None, None

        for epoch in range(self.n_epochs):
            self.model_t.train()
            for node_feat, adj, label in dataloader:
                node_feat = Variable(node_feat.cuda().type(self.Tensor))
                adj = Variable(adj.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                output = self.model_t(node_feat, adj, parc_type='schaefer', disease_type='MDD')
                loss = self.criterion_cls(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model_t.eval()
            with torch.no_grad():
                cls = self.model_t(node_feat_test, adj_test, parc_type='schaefer', disease_type='MDD')
                loss_test = self.criterion_cls(cls, test_label)
                y_hat = torch.max(cls, 1)[1]

                acc = accuracy_score(test_label.cpu(), y_hat.cpu())
                auc = roc_auc_score(test_label.cpu(), y_hat.cpu())
                cm = confusion_matrix(test_label.cpu(), y_hat.cpu())

                TP, FP, TN, FN = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]
                precision = TP / (TP + FP + 1e-8)
                recall = sensitivity = TP / (TP + FN + 1e-8)
                specificity = TN / (TN + FP + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}, "
                      f"Test Loss: {loss_test.item():.4f}, "
                      f"Test Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

                score = 0.5 * acc + 0.5 * auc
                if score > best_score:
                    best_score = score
                    best_acc = acc
                    best_auc = auc
                    y_true = test_label
                    y_pred = y_hat
                    print(">>> Model Updated (Best Score)")

                    # 可选：保存最佳模型
                    torch.save(self.model_t.state_dict(),
                               os.path.join(self.save_path, f"best_model_fold{self.fold_idx}.pth"))

        return best_acc, best_auc, y_true, y_pred


# ===============================
# ========== 主函数 ==============
# ===============================
def main():
    path = '/home/xinxu/Lehigh/Codes/lehigh_fmri/gpt_fmri/data_maml/maml_all.npy'
    pretrained_path = './checkpoints/pretrained_model.pth'  # 替换为你的预训练模型路径

    exp0 = ExP(0)
    total_data, total_label = exp0.get_data(path)

    idx_0 = np.where(total_label == 0)[0]
    idx_1 = np.where(total_label == 1)[0]
    selected_indices = np.concatenate([idx_0, idx_1])
    total_data = total_data[selected_indices]
    total_label = total_label[selected_indices]

    print(f"Label 0 count: {len(idx_0)}")
    print(f"Label 1 count: {len(idx_1)}")

    kf = KFold(n_splits=10, shuffle=True, random_state=88)

    all_true, all_pred = [], []
    metrics = {
        'acc': [], 'auc': [], 'f1': [], 'precision': [],
        'recall': [], 'sensitivity': [], 'specificity': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(total_data), 1):
        print(f"\n==== Fold {fold} ====")
        train_data = total_data[train_idx]
        test_data = total_data[test_idx]
        train_label = total_label[train_idx]
        test_label = total_label[test_idx]

        exp = ExP(fold_idx=fold, pretrained_path=pretrained_path)
        best_acc, best_auc, y_true, y_pred = exp.train([train_data, test_data, train_label, test_label])

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        TP, FP, TN, FN = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]

        metrics['acc'].append(accuracy_score(y_true, y_pred))
        metrics['auc'].append(roc_auc_score(y_true, y_pred))
        metrics['f1'].append(f1_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred))
        metrics['recall'].append(recall_score(y_true, y_pred))
        metrics['sensitivity'].append(TP / (TP + FN + 1e-8))
        metrics['specificity'].append(TN / (TN + FP + 1e-8))

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    print("\n=== Cross-Validation Results ===")
    for k, v in metrics.items():
        print(f"Average {k}: {np.mean(v):.4f}")

    cm = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()