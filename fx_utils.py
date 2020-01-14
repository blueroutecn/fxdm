# -*- coding: utf-8 -*-
from datetime import datetime as dt
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.utils.validation import _num_samples 


##a docorate function to estimate time consuming of the target fun 
def timecount():
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = dt.now()
            temp_result = func(*args, **kwargs)
            end_time = dt.now()
            time_pass = end_time-start_time
            print(("time consuming：" + str(np.round(time_pass.total_seconds()/60,2)) + "min").center(50, '='))
            return temp_result
        return wrapper
    return decorate

def is_monotonic(x):
    '''
    判断序列是否单调
    x list或者一维得np.array或者pd.Series
    '''
    dx = np.diff(x)
    return np.all(dx < 0).astype(int) or np.all(dx > 0).astype(int)

def check_unique(x):
    '''
    判断序列是否存在重复值
    x list或者一维得np.array或者pd.Series
    '''
    if len(np.unique(x))!=len(x):
        return False
    else:
        return True
    

def check_non_intersect(x,y):
    '''
    判断序列x和序列y是否存在交集
    x list或者一维np.array或者pd.Series
    y list或者一维得np.array或者pd.Series
    '''
    if len(set(x) & set(y)) != 0:
        print("存在交集:%s"%(set(x) & set(y)))
        return False
    else:
        return True

def dict_reverse(orig_dict):
    '''
    把一个map中的key和value反转，返回的map以之前的value作为key，并且每个value对应之前的一系列key组成的list
    '''
    reverse_dict = {}
    for key in orig_dict.keys():
        value = orig_dict.get(key)
        if value not in reverse_dict.keys():
            reverse_dict[value] = [key]
        else:
            temp_list = reverse_dict.get(value)
            temp_list.append(key)
            reverse_dict[value] = temp_list
    return reverse_dict


def psi(s1,s2):
    """
    用于计算psi s1和s2的长度必须相等
    s1: 对比序列1分箱之后每一箱的数据占比
    s2: 对比序列2分箱之后每一箱的数据占比
    """
    psi = 0
    s1 = list(s1)
    s2 = list(s2)
    if len(s1) != len(s2):
        print('序列s1和s2长度不等 请检查!')
        return None
    for i in range(len(s1)):
        ##处理下部分箱为0的情况
        if s2[i] == 0:
            s2[i]=0.000001
        if s1[i] == 0:
            s1[i]=0.000001
        p = ((s2[i]-s1[i])*(np.log(s2[i]/s1[i])))
        psi = psi+p
    return psi

def model_result_combine(model_result_dict,data_name):
    """
    把多个模型在多个数据集上预测的结果组成的dict中特定的数据集拿出来
    model_result_dict多个模型在多个数据集上的测试结果 形如{'model1':{'data1':{'predict':[],'true':[]},.....},.......}
    data_name特定数据集名称
    
    return:
    result_dict 转换后的dict{'model1':{'predict':[],'true':[]},.......}
    """
    result_dict = {}
    for model_name in model_result_dict.keys():
        result_dict[model_name] = {'predict':model_result_dict.get(model_name).get(data_name).get('predict'),
                                    'true':model_result_dict.get(model_name).get(data_name).get('true')}
    return result_dict

def data_sample(n_folds=5,frac=0.2,X=None,y=None,groups=None,oob=True,random_state=0):
    """
    把数据集划分成多份用于模型训练
    n_folds:如果是int类型 那么就做bootstrap抽样 抽取n_folds份
            如果是是包含split函数的类 那么就调用其split函数 取出valid部分
    frac:抽取的样本比例 只有到n_folds是int的时候有效 值在0到1之间
    X: X数据 
    y: Y数据
    groups: 如果根据自定义的分组情况进行CV 那么就需要这个参数 比如LeaveOneGroupOut这个数据切分方法
    oob: 是否需要同时返回out of bag的index
    random_state:随机种子
    
    return:
    index_list n个index array组成的list
    """
    train_index_list = []
    oob_index_list = []
    num_samples = _num_samples(X)
    np.random.seed(random_state)
    if isinstance(n_folds,int):
        if frac is None:
            batch_size = round(num_samples/n_folds)
        elif frac >= 0 and frac <=1:
            batch_size = round(num_samples*frac)
        else:
            raise ValueError("expect frac is a int object between 0 and 1 but got {0}".format(frac))
        for i in range(n_folds):
            train_index = np.random.choice(num_samples,batch_size,replace=True)
            oob_index = [i for i in range(num_samples) if i not in train_index]
            train_index_list.append(train_index)
            oob_index_list.append(oob_index)
    elif hasattr(n_folds,'split'):
        for fold_n, (train_index, valid_index) in enumerate(n_folds.split(X,y,groups)):
            train_index_list.append(valid_index)
            oob_index_list.append(train_index)
    if oob:
        return train_index_list,oob_index_list
    else:
        return train_index_list



