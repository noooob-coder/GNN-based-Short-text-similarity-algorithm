import csv
from torch.utils.data import DataLoader,Dataset
from fig import Config
config=Config()
class SDataItem(object):
    def __init__(self,input_text_1,input_text_2,out_put):
        self.input_1=input_text_1
        self.input_2=input_text_2
        self.out_put=out_put
class SDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.data=self.parse_file()
    def parse_file(self):
        with open(self.path, 'r') as f:
            reader = f.readlines()
            data = []
            for line in reader:
                line = "".join(line)
                line = line.strip('\n')
                line = line.split('\t')
                if 'sentence_A' in line[1]:
                    continue
                rst = SDataItem(input_text_1=line[1], input_text_2=line[2], out_put=float(line[3]))
                data.append(rst)
        return data
    def __getitem__(self, index):
        item=self.data[index]
        return item
    def __len__(self):
        return len(self.data)
