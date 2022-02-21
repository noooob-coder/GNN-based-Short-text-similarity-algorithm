import os
import pickle
import re
from collections import Counter
from functools import lru_cache
import warnings
from torchtext import data
import numpy as np
import torchtext
from nltk.tokenize import word_tokenize
from torchtext.data import Example,Dataset
import torch
from tqdm import tqdm
import os
from fig import Config
from torchtext.data import BucketIterator
config=Config()
from torchtext.vocab import Vectors, GloVe
class SentenceDataset(Dataset):
    def __init__(self,data,field):
        out : torch.DoubleTensor()=torchtext.data.Field(sequential=False,use_vocab=False,dtype=torch.double)
        fields=[('sentence_1',field),('sentence_2',field),('ou-t',out)]
        line=[]
        examples = []
        for item in data:
            line.append([item.input_1,item.input_2,item.out_put])
        for index in range(len(line)):
            #example中的out为浮点数
            examples.append(Example.fromlist([line[index][0],line[index][1],line[index][2]],fields))
        super(SentenceDataset, self).__init__(examples,fields)
    @staticmethod
    def sort_key(input):
        c = len(input.sentence_1) if len(input.sentence_1) > len(input.sentence_2) else len(input.sentence_2)
        return c

class Vocab(object):
    def __init__(self,dataset):
        self.tokenizer=word_tokenize
        self.data_path = './data/vocab_data.csv'
        self.all_words=self.collect_vocabs(dataset)
        self.vocab=self.vocab_df(dataset)
        #收集所有出现过的词
    def vocab_df(self,data):
        print('building vocab!')
        REVIEW = torchtext.data.Field(sequential=True, tokenize=self.tokenizer, lower=True, include_lengths=True)
        data=SentenceDataset(data,REVIEW)
        # print(data.examples[0].sentence_1)
        # print(data.examples[0].sentence_2)
        REVIEW.build_vocab(data,  # 建词表是用训练集建，不要用验证集和测试集
                           max_size=len(self.all_words.keys()),  # 单词表容量
                           vectors='glove.6B.300d',  # 还有'glove.840B.300d'已经很多可以选
                           unk_init=torch.Tensor.normal_  # 初始化train_data中不存在预训练词向量词表中的单词
                           )
        self.REVIEW = REVIEW
        return REVIEW.vocab
    def data_load(self,data):
        text_data=SentenceDataset(data,self.REVIEW)
        data_iter=BucketIterator(
            dataset=text_data,
            batch_size=config.batch_size,
            repeat=False,
            sort_key=lambda x:text_data.sort_key(x),
            device=config.device,
            sort_within_batch=False,
            sort=True,
        )
        return data_iter
    def collect_vocabs(self,dataset):
        all_words = Counter()
        print('读取data：')
        line=[]
        for item in tqdm(dataset):
            token1=word_tokenize(item.input_1.lower())
            token2=word_tokenize(item.input_2.lower())
            all_words.update(token1)
            all_words.update(token2)
            line.append(item.input_1)
            line.append(item.input_2)
        a=os.path.exists(self.data_path)
        if not a :
            os.mknod(self.data_path)
        with open(self.data_path, 'w', newline='') as file:
            file.truncate()
            for row in line:
                file.write(row+'\n')
        return all_words