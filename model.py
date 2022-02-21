from torch.nn.modules import Module
from torch import nn
import torch
import dgl
import torch.nn.functional as F
from fig import Config
from dgl.nn import SAGEConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import WeightAndSum
import warnings
from dgl.nn import Set2Set

warnings.filterwarnings("ignore")
config=Config()
class MutiGraph(Module):
    def __init__(self):
        super(MutiGraph, self).__init__()
        self.head_weighting = nn.Sequential(
            nn.Linear(300, 1),
            nn.Sigmoid()
        )
        self.total_head_num=config.num_heads
        self.relu = nn.ReLU()
        self.sage = SAGEConv(300, out_feats=300, feat_drop=config.sage_drop, aggregator_type='gcn', bias=True)
        self.gatconv=GATConv(300,out_feats=300,feat_drop=config.feat_drop,attn_drop=config.attn_drop,num_heads=config.num_heads,allow_zero_in_degree=True)
    def graph_sage(self, g):
        h = g.ndata['sage']
        h = self.sage(g, h)
        g.ndata['sage'] = h
    def graph_gat(self, g):
        h = g.ndata['gat']
        h = self.gatconv(g, h)
        g.ndata['gat'] = h
    def sum_sage_attention(self,g,g1):
        attention=g.ndata['gat']
        attention=attention.view(attention.size()[0],self.total_head_num,300)
        sage = g1.ndata['sage'].view(attention.size()[0],1,attention.size()[2])
        attention=torch.cat((attention,sage),dim=1)
        weight = self.head_weighting(attention)
        attention=torch.mul(attention,weight)
        out=torch.sum(attention,dim=1)
        g.ndata['embeding']=out
    def forward(self,g,g1):
        base=g.ndata['embeding']
        g1.ndata['sage']=g1.ndata['embeding']
        g.ndata['gat']=g.ndata['embeding']
        self.graph_gat(g)
        self.graph_sage(g1)
        self.sum_sage_attention(g,g1)
        end=g.ndata['embeding']
        g.ndata['embeding']=torch.add(base,end)
        g1.ndata['embeding']=g.ndata['embeding']
        # print(base.size())
        # print(end.size())
        return g,g1
class Pool_Graph(Module):
    def __init__(self):
        super(Pool_Graph,self).__init__()
        self.weight_sum = WeightAndSum(300)
    def forward(self,g):
        rst = self.weight_sum(g, g.ndata['embeding'])
        batch_num=g.batch_size
        rst=rst.view(batch_num,300)  #为每个图的embeding头数
        return rst
class Graph_build(Module):
    def __init__(self):
        super(Graph_build, self).__init__()
        self.similarity_weight = torch.nn.Parameter(torch.randn(300).double(), requires_grad=True)
    def build_edg(self,g,adj):
        edg = adj.nonzero()
        node_set_1, node_set_2 = edg.split([1, 1], dim=1)
        node_set_1 = node_set_1.view(node_set_1.size()[0])
        node_set_2 = node_set_2.view(node_set_2.size()[0])
        g.add_edges(node_set_1, node_set_2)
        return g
    def adj_sparsification(self,adj):
        base_ajd = adj
        zero = torch.zeros(adj.size()).double()
        zero = zero.to(config.device)
        adj = torch.triu(adj, diagonal=1)
        top_k = config.top_k
        num = adj.numel()
        if num > top_k:
            flat = adj.view(num)
            t, _ = torch.topk(flat, top_k, sorted=True)
            flag = t[-1]
            flag.to(config.device)
            new_adj = torch.where(base_ajd >= 0.1, base_ajd, zero)
            return new_adj
        else:
            return base_ajd
    def similarity_learn(self,g):
        ebd = g.ndata['embeding']
        ebd = F.relu(torch.mul(ebd, self.similarity_weight))
        adj = torch.mm(ebd, ebd.t())
        adj=self.adj_sparsification(adj)
        return adj
    def build_graph(self,g_list):
        g_list=dgl.unbatch(g_list)
        g_new=[]
        for g in g_list:
            adj=self.similarity_learn(g)
            g=self.build_edg(g,adj) #构建边
            g=dgl.add_self_loop(g)
            g_new.append(g)
            # print(g.ndata['embeding'].size())
        total_g=dgl.batch(g_new)
        return total_g
    def forward(self,g):
        g=self.build_graph(g)
        return g
class Score_construct(Module):
    def __init__(self):
        super(Score_construct, self).__init__()
        self.softlayer = nn.Sequential(
            nn.Softmax(dim=1),
            nn.ReLU(),
            nn.Linear(600, 1)
        )
    def soft(self,g1,g2,len):
        g = torch.cat((g1, g2), 1)
        rst = self.score(g)
        rst = rst.view(len)
        return rst
    def pearson(self,g1,g2):
        mean_1 = torch.mean(g1, dim=1)
        mean_2 = torch.mean(g2, dim=1)
        mean_1 = torch.unsqueeze(mean_1, 1)
        mean_2 = torch.unsqueeze(mean_2, 1)
        g1 = g1.sub(mean_1)
        g2 = g2.sub(mean_2)
        norm = torch.sum(g1 * g2, dim=1)
        sq_1 = torch.sqrt(torch.sum(g1 ** 2, dim=1))
        sq_2 = torch.sqrt(torch.sum(g2 ** 2, dim=1))
        rst = torch.div(norm, torch.mul(sq_1, sq_2))
        rst = torch.mul(rst, 5)
        return rst
    def forward(self,g1,g2,len):
        if config.socre_method=='pearson':
            rst=self.pearson(g1,g2)
        if config.socre_method=='soft':
            rst = self.soft(g1, g2,len)
        return rst  # rst为batch数*头数
class Base_graph(object):
    def graph(self,ebd):
        ebd = torch.transpose(ebd, 0, 1)
        g_list = []
        for sentence in ebd:
            g = dgl.DGLGraph().to(config.device)
            g.add_nodes(num=sentence.size()[0])
            g.ndata['embeding'] = sentence
            g_list.append(g)
        total_g = dgl.batch(g_list)
        return total_g
class MyModel(Module):
    def __init__(self,vocab_model):
        super(MyModel, self).__init__()
        weight_matrix = vocab_model.vocab.vectors
        self.embeding=nn.Embedding(len(vocab_model.vocab),300)
        self.embeding.weight.data.copy_(weight_matrix)
        self.graph_pooling=Pool_Graph()
        self.total_head_num=config.num_heads
        self.linear=nn.Linear(self.total_head_num,1)
        self.relu = nn.ReLU()
        self.graph_embedding_1=MutiGraph()
        self.build_graph_base=Base_graph()
        self.build_graph_1 = Graph_build()
        self.build_graph_2 = Graph_build()
        self.score_construct=Score_construct()
        self.graph_embedding_2 = MutiGraph()
        self.graph_embedding_3 = MutiGraph()
        self.graph_embedding_4 = MutiGraph()
        self.sts = Set2Set(300, 2, 1)
            #获取adj中的前k大元素并保留位置
    def full(self,s):
        # g1为topk g2为全链接
        g1,g2 = self.build_graph(s)
        g1,g2 = self.graph_embedding_1(g1,g2)
        g1, g2 = self.graph_embedding_2(g1, g2)
        return g1
    def notfull(self,g):
        g1=self.build_graph_2(g)
        g2 = self.build_graph_1(g)
        g1, g2 = self.graph_embedding_1(g1, g2)
        g1 = self.build_graph_2(g)
        g2 = self.build_graph_1(g)
        g1, g2 = self.graph_embedding_2(g1, g2)
        return g1
    def forward(self,data):
        self.device=data.sentence_1[0].device
        sentence_1 = self.embeding(data.sentence_1[0])
        sentence_2=self.embeding(data.sentence_2[0])
        g1=self.build_graph_base.graph(sentence_1)
        g2 = self.build_graph_base.graph(sentence_2)
        #进入网络结构
        if config.build=='full':
            g1=self.full(g1)
            g2=self.full(g2)
        if config.build=='notfull':
            g1 = self.notfull(g1)
            g2 = self.notfull(g2)
        pool_rst=self.graph_pooling(g1)  #图2的最终表示
        pool_rst_2=self.graph_pooling(g2)
        similarity=self.score_construct(pool_rst,pool_rst_2,len(data))
        return similarity