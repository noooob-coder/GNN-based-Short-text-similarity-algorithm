from data_load import SDataset,SDataItem
from fig import Config
from vocab import Vocab
import torch.optim as optim
import torch
import datetime
from loss import MyLoss
import time
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from model import MyModel
from prettytable import PrettyTable
from eval import Eval
from utils import wirte_log,model_save,model_load,to_dvice,lr_down

lr_list=[]
def train():
    global config
    config = Config()
    train_data = SDataset(config.train_path)
    test_data = SDataset(config.test_path)
    total_data = test_data.data + train_data.data
    vocabe_model = Vocab(total_data)  # 设置词向量模型
    train_set = vocabe_model.data_load(train_data.data)
    test_set = vocabe_model.data_load(test_data.data)
    eva=Eval(test_set)
    set_lsit = [train_set, test_set]
    to_dvice(set_lsit)
    torch.cuda.set_device(config.device)
    # train_set=DataLoader(dataset=train_data,batch_size=config.batch_size,shuffle=True,num_workers=1,)
    model = MyModel(vocabe_model)
    model.to(config.device)
    if config.load_flag=='y':
        print('读取预训练模型')
        model=model_load(model)
    lr=config.lr
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    lossfunc = torch.nn.MSELoss(reduce=True, size_average=True)
    for i in range(config.epoch):
        for index, data in enumerate(train_set):
            model.train()
            data.sentence_1[0].to(config.device)
            data.sentence_2[0].to(config.device)
            data.out[0].to(config.device)
            right=data.out
            output=model(data)
            loss=lossfunc(right,output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            show='{},第{}个epoch，第{}步，loss为:{}，acc为:{}'.format(config.datasetname,i + 1, index, loss,eva.acc)
            print(show)
            wirte_log(show)
        model_save(model)
        eva.test(model)

train()