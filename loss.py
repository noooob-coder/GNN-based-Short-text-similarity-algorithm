import torch
import torch.nn as nn
from scipy.stats import pearsonr
class MyLoss(nn.Module):
    def __init__(self,lr):
        super(MyLoss,self).__init__()
        self.lr=lr
    def forward(self,out_1,out_2):
        mean_1 = torch.mean(out_1.float())
        mean_2 = torch.mean(out_2.float())
        g1=out_1-mean_1
        g2=out_2-mean_2
        norm = torch.sum(g1 * g2)
        sq_1 = torch.sqrt(torch.sum(g1 ** 2))
        sq_2 = torch.sqrt(torch.sum(g2 ** 2))
        rst = norm / (sq_1 * sq_2)
        rst=1-rst
        return rst*self.lr
