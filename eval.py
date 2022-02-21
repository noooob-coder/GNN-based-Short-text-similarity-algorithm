from fig import Config
import tqdm
import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from utils import wirte_log
import numpy as np
from sklearn.metrics import mean_squared_error
def MSE(y, t):
    return mean_squared_error(y,t)

class Eval(object):
    def __init__(self,test_set):
        self.test_set=test_set
        self.max_pccs=0
        self.max_spear=0
        self.max_mse=100
        self.acc = (self.max_pccs, self.max_spear, self.max_mse)
        self.config=Config()
    def eva(self,right,out):
        right=right.cpu().numpy()
        out=out.cpu().detach().numpy()
        pccs=pearsonr(right,out)[0]
        spear=spearmanr(right,out)[0]
        mse=MSE(right,out)
        if pccs > self.max_pccs:
            self.max_pccs=pccs
        if spear >self.max_spear:
            self.max_spear=spear
        if mse<self.max_mse:
            self.max_mse=mse
        return pccs,spear,mse
    def test(self,model):
        global config
        device=self.config.device
        # train_set=DataLoader(dataset=train_data,batch_size=config.batch_size,shuffle=True,num_workers=1,)
        model.eval()
        with torch.no_grad():
            model.to(device)
            rst_list=[]
            out_list=[]
            for index,data in tqdm.tqdm(enumerate(self.test_set)):
                data.sentence_1[0].to(device)
                data.sentence_2[0].to(device)
                data.out[0].to(device)
                rst_line=data.out
                out_line=model(data)
                rst_list.append(rst_line)
                out_list.append(out_line)
            rst=torch.cat([x for x in rst_list],0)
            out=torch.cat([x for x in out_list],0)
            print('computing!')
            a,b,c=self.eva(rst,out)
            self.acc = (self.max_pccs, self.max_spear, self.max_mse)
            item='现有的结果为，pearson：{}，spearman：{}，mse：{}'.format(a,b,c)
            wirte_log(item)
            item = '最好的结果为，pearson：{}，spearman：{}，mse：{}'.format(self.max_pccs, self.max_spear, self.max_mse)



