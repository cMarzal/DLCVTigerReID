from torch import nn
import torch
from torch.autograd import Variable


class TripletLoss(object):
    def __init__(self, margin=0.3):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def get_dist(imgs, labels):
    dist_mat = euclidean_dist(imgs, imgs)

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    pr3 = dist_mat[is_pos].contiguous()
    pr4 = dist_mat[is_neg].contiguous()
    if(len(pr3) % N != 0):
        rem = len(pr3) % N
        pr3 = pr3[0:(len(pr3)-rem)]

    if (len(pr4) % N != 0):
        rem = len(pr4) % N
        pr4 = pr4[0:(len(pr4) - rem)]

    dist_ap, relative_p_inds = torch.max(
        pr3.view(N, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(
        pr4.view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def local_dist(x, y):

    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def shortest_dist(dist_mat):
    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist


def get_dist_local(local_feat,labels):

    dist_mat = local_dist(local_feat, local_feat)

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    pr3 = dist_mat[is_pos].contiguous()
    pr4 = dist_mat[is_neg].contiguous()
    if len(pr3) % N != 0:
        rem = len(pr3) % N
        pr3 = pr3[0:(len(pr3) - rem)]

    if len(pr4) % N != 0:
        rem = len(pr4) % N
        pr4 = pr4[0:(len(pr4) - rem)]

    dist_ap, relative_p_inds = torch.max(
        pr3.view(N, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(
        pr4.view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an
