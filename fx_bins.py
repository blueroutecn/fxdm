# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:51:09 2020

@author: doublewen
"""
import pandas as pd
import numpy as np
from fx_utils import is_monotonic,psi,timecount,check_unique,dict_reverse
import warnings
warnings.filterwarnings("ignore")


"""
分箱、WOE转换以及IV相关函数
"""
##---------------------------------无监督分箱--------------------------------------------
def quantile_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于等频分箱 返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    
    '''
    print('正在进行变量{0}的等频分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##这里用1:-1的原因是10分箱只需要9个cut off point就可以了
    cp = [binning_series.quantile(i) for i in np.linspace(0,1,max_interval+1)[1:-1]]
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

def distance_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于等距分箱返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的等距分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##这里用1:-1的原因是10分箱只需要9个cut off point就可以了
    cp = list(np.linspace(binning_series.min(),binning_series.max(),max_interval+1,endpoint=True)[1:-1])
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

def mix_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于混合分箱返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    混合分箱的存在是为了防止异常值的存在对等距分箱的影响 在头尾进行等频率的分箱 然后剩下的部分用等距分箱
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的混合分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##混合分箱
    quantile_cp = [binning_series.quantile(i) for i in np.linspace(0,1,max_interval+1)[1:-1]]
    distance_cp = list(np.linspace(quantile_cp[0],quantile_cp[-1],max_interval-1,endpoint=True)[1:-1])
    cp = [quantile_cp[0]] + distance_cp + [quantile_cp[-1]]
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

##---------------------------------有监督决策树分箱--------------------------------------------
from sklearn.tree import DecisionTreeClassifier
def tree_binning(data, var, label, special_attribute=[],treeClassifier=DecisionTreeClassifier()):
    """
    用于决策树分箱 不允许存在缺失值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    treeClassifier: sklearn的DecisionTreeClassifier类或者有类似功能的其他类
    
    return:
    cp: 分箱切分的节点
    """
    print('正在进行变量{0}的决策树分箱'.format(var))
    ##从DataFrame里面取出对应的数据
    binning_data = data[[var, label]].loc[~data[var].isin(special_attribute)]
    ##判断是否存在缺失值
    if (np.sum(binning_data[var].isna())>0) or (np.sum(binning_data[label].isna())>0):
        raise ValueError("detect nan values in {0}".format([var,label]))
    ##进行决策树拟合
    treeClassifier.fit(X=binning_data[var].values.reshape(-1, 1),y=binning_data[label].values.reshape(-1, 1))
    cp = sorted(treeClassifier.tree_.threshold[treeClassifier.tree_.threshold != -2])
    ##处理没有找到任何可能分箱的情况
    if len(cp) == 0:
        raise ValueError("detect empty cp for {0} in tree_binning".format([var,label]))
    return cp

##---------------------------------有监督卡方分箱--------------------------------------------
#-----------------辅助函数1 初始化数据分箱-----------
def SplitData(df, col, numOfSplit, special_attribute=[]):
    """
    在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 
    splitPoint： 初始化数据分箱的节点
    """
    df2 = df.copy()
    if len(special_attribute) > 0:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = int(N / numOfSplit)
    splitPointIndex = [i*n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))

    aa = pd.Series(splitPoint)
    if (aa[0] == 0.0) & (aa.shape[0] == 1):
        numOfSplit = 1000
        n = int(N / numOfSplit)
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(df2[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
    else:
        pass
    return splitPoint
