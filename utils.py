from typing import List
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score


def adj_to_edge_index_attr(adj_matrix):
    #adj_matrix's shape: [Batch, time_series, roi_number]
    edge_index_list = []
    edge_attr_list = []
    num_nodes = adj_matrix.shape[1]

    # For each batch
    for i in range(adj_matrix.shape[0]):
        # Get the adjacency matrix for the current batch
        adj = adj_matrix[i]

        # Get the indices and values of the non-zero elements of the adjacency matrix
        nonzero = torch.nonzero(adj)
        values = adj[nonzero[:, 0], nonzero[:, 1]]

        # Add the batch index to the indices
        indices = torch.cat([nonzero[:, 1:], nonzero[:, :1].repeat(1, 2)], dim=1)
        indices[:, 0] += i * num_nodes
        indices[:, 1] += i * num_nodes

        # Append the indices and values to the lists
        edge_index_list.append(indices)
        edge_attr_list.append(values)

    # Concatenate the indices and values
    edge_index = torch.cat(edge_index_list, dim=0).t().contiguous()
    edge_attr = torch.cat(edge_attr_list, dim=0)

    return edge_index, edge_attr


#tensor -> tensor
def edge2adj(edge_index, edge_attr):
    num_nodes = edge_index.max().item() + 1
    adj = torch.sparse_coo_tensor(edge_index, edge_attr, (num_nodes, num_nodes))
    adj = adj.to_dense()
    adj.fill_diagonal_(0)
    return adj






def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return new_xs, y



def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    mix_xs, mix_nodes, mix_ys = [], [], []
    for t_y in y.unique():
        idx = y == t_y
        t_mixed_x, t_mixed_nodes, _, _, _ = continus_mixup_data(x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)
        mix_ys.append(y[idx])
    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_cluster_loss(matrixs, y, intra_weight=2):
    y_1 = y[:, 1]
    y_0 = y[:, 0]
    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0
    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))
    return loss




def inner_loss(label, matrixs):
    loss = 0
    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))
    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))
    return loss




def intra_loss(label, matrixs):
    a, b = None, None
    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)
    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0



def count_params(model: nn.Module, only_requires_grad: bool = False):
    "count number trainable parameters in a pytorch model"
    if only_requires_grad:
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params





def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)
    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def R2_score(output: torch.Tensor, target: torch.Tensor) -> List[float]:
    r2 = r2_score(target.cpu().detach().numpy().flatten(), output.cpu().detach().numpy().flatten())
    return r2

def R2_score_list(output, target):
    r2 = r2_score(target, output)
    return r2




class WeightedMeter:
    def __init__(self, name: str = None):
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0

    def update(self, val: float, num: int = 1):
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count
        self.val = val

    def reset(self, total: float = 0, count: int = 0):
        self.count = count
        self.sum = total
        self.avg = total / max(count, 1)
        self.val = total / max(count, 1)





class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    @property
    def val(self) -> float:
        return self.history[self.current]

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val
        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val





class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    adj = np.random.rand(1000, 100, 100)