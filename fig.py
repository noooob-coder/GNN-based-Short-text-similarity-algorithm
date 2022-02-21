import torch
class Config(object):
    def __init__(self):
        self.train_path = './data/SICK_train.txt'
        self.test_path = './data/SICK_test_annotated.txt'
        self.dev_path = './data/my_data.txt'
        self.batch_size=1
        self.device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.top_k=5
        self.feat_drop=0.2
        self.attn_drop=0.2
        self.sage_drop=0.2
        self.sage_layers=2
        self.num_heads = 3
        self.layers=2
        self.epoch=300
        self.load_flag='y'
        self.lr=0.01
        self.loss='my'
        self.build='notfull'
        self.socre_method='pearson'
        self.early_stop=20
        self.log_path='./log.txt'
        self.datasetname='sts-b {} head {} layer {} topk'.format(self.num_heads,self.layers,self.top_k)