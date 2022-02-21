from fig import Config
import torch
config=Config()
def wirte_log(item):
    global config
    with open(config.log_path, 'a') as f:
        f.write('-'*50)
        f.write('\n')
        f.write(config.datasetname)
        f.write('\n')
        f.write(item)
        f.write('\n')
        f.write('-'*50)
        f.write('\n')
def model_save(model):
    print('-' * 50)
    print('正在保存model')
    torch.save(model.state_dict(), './model.bt')
    print('保存成功')
    print('-'*50)
def model_load(model):
    print('-' * 50)
    print('正在读取model')
    model.load_state_dict(torch.load('./model.bt'))
    print('读取成功')
    print('-' * 50)
    return model
def to_dvice(list):
    print('正在将数据加载到显卡')
    for i in list:
        for data in i:
            data.sentence_1[0].to(config.device)
            data.sentence_2[0].to(config.device)
            data.out[0].to(config.device)
def lr_down(lr_list,item):
    if len(lr_list)>=config.early_stop:
        lr_list=lr_list[1:]
        lr_list.append(item)
    else:lr_list.append(item)
    return lr_list
