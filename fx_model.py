# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from fx_utils import timecount
from fx_evaluation import plot_ks_curve,plot_roc_curve
from bayes_opt import BayesianOptimization
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn2pmml.decoration import ContinuousDomain,CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import gen_features
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.impute import SimpleImputer

def feature_rank_calculator_lgb(X,y,numeric_feature,category_feature,params_dict={}):
    '''
    计算单一数据集在lgb模型下面的特征重要
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    numeric_feature: 需要处理的数值型变量
    category_feature: 需要处理的类别型变量
    params_dict: lightgbm的模型参数
    
    return:
    importance_score: 特征重要度 pandas.Series对象
    '''
    if len(category_feature) == 0:
        data = lgb.Dataset(X[numeric_feature+category_feature], label=y)
    else:
        data = lgb.Dataset(X[numeric_feature+category_feature], label=y,categorical_feature=category_feature) 
    model = lgb.train(params_dict,data,verbose_eval=None)
    importance_score = pd.Series(model.feature_importance(importance_type='gain', iteration=None),index=numeric_feature+category_feature)
    return importance_score

@timecount()
def feature_rank_split_calculator_lgb(X,y,params_dict_list,numeric_feature,category_feature,groups=None,use_validation=True,split_generator=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)):
    '''
    通过多个lgb模型和多个数据集划分方案计算特征重要度情况
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    params_dict_list:传给各个lgb模型的参数
    numeric_feature: 需要处理的数值型变量
    category_feature: 需要处理的类别型变量
    groups:如果一些自定义的分组情况进行CV 那么就需要这个参数 比如LeaveOneGroupOut这个数据切分方法
    use_validation:通过split中的train还是valid部分进行特征计算
    split_generator:继承至SKlearn的一系列CV生成函数
    '''
    print('开始计算特征重要度'.center(50, '='))
    final_importance_score_list = []
    col_name_list= []
    ##两层循环进行estimator和数据集的遍历
    for estimator_n,params_dict in enumerate(params_dict_list):
        for fold_n, (train_index, valid_index) in enumerate(split_generator.split(X,y,groups)):
            print("分类器:%s 折数:%s 数据量大小:%s 坏用户比率:%s"% (estimator_n, fold_n,len(valid_index),np.sum(y[valid_index])/len(valid_index)))
            if groups is not None:
                col_name_list.append(str(estimator_n)+'_'+str(fold_n)+'_'+str(groups[valid_index].iloc[0]))
            else:
                col_name_list.append(str(estimator_n)+'_'+str(fold_n))
            
            if use_validation:
                temp_x = X.iloc[valid_index,:]
                temp_y = y.iloc[valid_index]
            else:
                temp_x = X.iloc[train_index,:]
                temp_y = y.iloc[train_index]
            temp_importance_score = feature_rank_calculator_lgb(temp_x,temp_y,numeric_feature,category_feature,params_dict)
            final_importance_score_list.append(temp_importance_score)
    importance_table_orig = pd.concat(final_importance_score_list,axis = 1)
    importance_table_orig.columns = col_name_list
    importance_table_stats = pd.DataFrame({'mean':importance_table_orig.mean(axis=1),'std':importance_table_orig.std(axis=1),'max':importance_table_orig.max(axis=1),'min':importance_table_orig.min(axis=1)},index=importance_table_orig.index)
    
    return importance_table_orig,importance_table_stats




def gbm_cv_evaluate(X,y,total_features,category_features,cv,groups=None,X_test=None,y_test=None,params_dict=None):
    '''
    单个light模型的CV评估的结果
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    total_features: 入模所有特征 list
    category_features: 入模类别特征 list
    cv: 数据集切分方法
    groups: 数据分组
    X_test:测试X数据 可不指定
    y_test:测试y数据 可不指定
    params_dict: 模型参数
    
    return:
    detail_result: 每一折cv的结果 pandas.DataFrame对象
    statistic_result 最终各指标的结果 dict对象
    '''
    valid_auc_list = []
    valid_ks_list = []
    train_auc_list = []
    train_ks_list = []
    if X_test is None:
        test_auc_list = np.nan
        test_ks_list = np.nan
    else:
        test_auc_list = []
        test_ks_list = []
        
    best_iteration_list = []
    
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):
        
        ##取出当前轮使用的的训练数据和验证数据
        X_valid,y_valid = X.iloc[valid_index][total_features],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index][total_features],y.iloc[train_index]
        if len(category_features) == 0:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)  
        else:
            train_data = lgb.Dataset(X_train, label=y_train,categorical_feature=category_features)
            valid_data = lgb.Dataset(X_valid, label=y_valid,categorical_feature=category_features)  
        ##模型训练
        model = lgb.train(params_dict,train_data,num_boost_round=20000,\
                          valid_sets = [valid_data],verbose_eval=None,early_stopping_rounds=100)
        ##模型预测
        y_pred_train = model.predict(X_train,num_iteration=model.best_iteration)
        y_pred_valid = model.predict(X_valid,num_iteration=model.best_iteration)
        ##结果评估
        train_auc = plot_roc_curve(y_train,y_pred_train)
        train_ks = plot_ks_curve(y_pred_train,y_train, is_score=False, n=10)
        valid_auc = plot_roc_curve(y_valid,y_pred_valid)
        valid_ks = plot_ks_curve(y_pred_valid,y_valid, is_score=False, n=10)
        ##结果记录
        train_auc_list.append(train_auc)
        train_ks_list.append(train_ks)
        valid_auc_list.append(valid_auc)
        valid_ks_list.append(valid_ks)
        if X_test is not None:
            y_pred_test = model.predict(X_test[total_features],num_iteration=model.best_iteration)
            test_auc = plot_roc_curve(y_test,y_pred_test)
            test_ks = plot_ks_curve(y_pred_test,y_test, is_score=False, n=10)
            test_auc_list.append(test_auc)
            test_ks_list.append(test_ks)
        best_iteration_list.append(model.best_iteration)
        
    detail_result = pd.DataFrame(data={'test_auc':test_auc_list,
                                          'test_ks':test_ks_list,
                                          'valid_auc':valid_auc_list,
                                          'valid_ks':valid_ks_list,
                                          'train_ks':train_ks_list,
                                          'train_auc':train_auc_list,
                                          'best_iteration':best_iteration_list
                                          })
    statistic_result = { 'train_auc_mean':np.mean(train_auc_list),
                        'train_auc_std':np.std(train_auc_list),
                        'train_ks_mean':np.mean(train_ks_list),
                        'train_ks_std':np.std(train_ks_list),
                        'valid_auc_mean':np.mean(valid_auc_list),
                        'valid_auc_std':np.std(valid_auc_list),
                        'valid_ks_mean':np.mean(valid_ks_list),
                        'valid_ks_std':np.std(valid_ks_list),
                        'test_auc_mean':np.mean(test_auc_list),
                        'test_auc_std':np.std(test_auc_list),
                        'test_ks_mean':np.mean(test_ks_list),
                        'test_ks_std':np.std(test_ks_list)
                   }
    print('train AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(train_auc_list), np.std(train_auc_list)),\
          'train KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(train_ks_list), np.std(train_ks_list)))
    print('valid AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(valid_auc_list), np.std(valid_auc_list)),\
          'valid KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(valid_ks_list), np.std(valid_ks_list)))
    if X_test is not None:
        print('test AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(test_auc_list), np.std(test_auc_list)),\
              'test KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(test_ks_list), np.std(test_ks_list)))
    print('best_iteration:',best_iteration_list)
    return detail_result,statistic_result



def feature_selector(feature_rank,params_dict,X_train,y_train,var_type_dict,X_test=None,y_test=None,cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),groups=None,rounds=100,step=1,auc_diff_threshold=0):
    '''
    逐步加入变量的stepwise特征筛选函数 使用lightgbm模型
    feature_rank: 原本的特征重要度排序 Series
    params_dict: LGBMClassifier的相应参数
    X_train: 训练X
    y_train: 训练y
    var_type_dict: 包含数据中的各变量类型
    X_test: 测试X 可不给
    y_test: 测试y 可不给
    cv: 数据集切分方法 默认为5折StratifiedKFold
    groups: cv所用的参数
    rounds: 总共筛选多少轮
    step: 每轮加入多少个变量
    auc_diff_threshold: auc有多少提升才会被选入到其中
    
    return:
    detail_map 每一轮cv的细节内容构成的map key为roundx x为论数
    feature_selected_statistic 逐步选入模型的变量的每一步评估结果
    '''
    print("开始特征筛选".center(50, '='))
    each_round_start = range(0,rounds*step,step)
    category_var = var_type_dict.get('category_var')
    auc_initial = 0.6
    feature_selected = []
    feature_selected_num = []
    feature_selected_statistic = pd.DataFrame()
    detail_map = {}
    for i in each_round_start:
        print('*************rounds: %d****************'%(i/step+1))
        temp_detail_map = {}
        importanceFeature_add = feature_rank[i:i+step]
        temp_feature = feature_selected + list(importanceFeature_add.index)
        temp_category = [i for i in temp_feature if i in category_var]
        detail_result,statistic_result = gbm_cv_evaluate(X_train,y_train,temp_feature,temp_category,cv,groups,X_test,y_test,params_dict)
        print('入模型特征数:'+str(len(temp_feature)))
        print('当前轮数考察特征:',list(importanceFeature_add.index))        
        
        temp_detail_map['auc_threshold'] = auc_initial
        temp_detail_map['feature_initial'] = feature_selected
        temp_detail_map['feature_add'] = importanceFeature_add
        temp_detail_map['feature_used'] = temp_feature
        temp_detail_map['category_feature_used'] = temp_category
        temp_detail_map['detail_result'] = detail_result
        temp_detail_map['statistic_result'] = statistic_result
        temp_detail_map['is_delete'] = (statistic_result['valid_auc_mean']-auc_initial) < auc_diff_threshold
        
        if (statistic_result['valid_auc_mean']-auc_initial) < auc_diff_threshold:
            print('删除当前批次的入模特征')
            continue
        feature_selected = feature_selected + list(importanceFeature_add.index)
        feature_selected_num.append(len(feature_selected))
        auc_initial = statistic_result['valid_auc_mean']
        detail_map['round'+str(int(i/step+1))] = temp_detail_map
        feature_selected_statistic =  pd.concat([feature_selected_statistic,pd.DataFrame(temp_detail_map['statistic_result'],index=['round'+str(int(i/step+1))])]) 
    return detail_map,feature_selected_statistic




def bayes_parameter_opt_lgb(X, y, category_feature, init_round=15, opt_round=25, n_folds=5, random_seed=0, n_estimators=10000,
                            learning_rate=0.05):
    '''
    贝叶斯超参数筛选 lightgbm版
    X: 训练X
    y: 训练y
    category_feature: 类别特征（）
    init_round: 最开始随机搜索的次数
    opt_round: 贝叶斯优化搜搜的次数
    n_folds: CV的折数
    random_seed: 随机种子
    n_estimators: 模型树个数上限
    learning_rate: 学习率
    
    return:
    最优的参数
    
    
    调用例子：
    lgb_opt_params = bayes_parameter_opt_lgb(X, y,category_feature,init_round=5, opt_round=15, n_folds=5, random_seed=0, n_estimators=10000,
                                     learning_rate=0.05)
    '''
    # prepare data
    train_data = lgb.Dataset(data=X, label=y,categorical_feature=category_feature, free_raw_data=False)

    # parameters
    def lgb_eval(num_leaves, max_depth, reg_alpha, reg_lambda, min_split_gain,min_child_weight,max_bin):
        params = {'application': 'binary', 'num_iterations': n_estimators, 'learning_rate': learning_rate, 
                  'early_stopping_round': 100, 'metric': 'auc'}
        params["nthread"] = 4
        params["num_leaves"] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params["silent"] = -1
        params["verbose"] = -1
        params['max_bin'] = int(round(max_bin))
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (12, 50),
                                            'max_bin': (50, 300),
                                            'max_depth': (2, 5),
                                            'reg_alpha': (0, 4),
                                            'reg_lambda': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 70)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return lgbBO.max




def bayes_parameter_opt_xgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=0, n_estimators=10000,
                            learning_rate=0.05):
    '''
    贝叶斯超参数筛选 xgboost版
    X: 训练X
    y: 训练y
    init_round: 最开始随机搜索的次数
    opt_round: 贝叶斯优化搜搜的次数
    n_folds: CV的折数
    random_seed: 随机种子
    n_estimators: 模型树个数上限
    learning_rate: 学习率
    
    return:
    最优的参数
    
        
    调用例子:
    xgb_opt_params = bayes_parameter_opt_xgb(X, y, init_round=5, opt_round=15, n_folds=5, random_seed=0, n_estimators=10000,
                                     learning_rate=0.1)
    '''
    # prepare data
    train_data =xgb.DMatrix(data=X, label=y)

    # parameters
    def xgb_eval(max_depth, reg_alpha, reg_lambda, gamma,min_child_weight,subsample,colsample_bytree):
        params = {'application': 'binary:logistic','learning_rate': learning_rate,'metric': 'auc'}
        params["nthread"] = 4
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['gamma'] = gamma
        params['min_child_weight'] = min_child_weight
        params['eval_metric'] = 'auc'
        params["silent"] = 1
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        cv_result = xgb.cv(params, train_data,num_boost_round=n_estimators, nfold=n_folds, seed=random_seed, verbose_eval=100,early_stopping_rounds=100,
                       metrics=['auc'],maximize=True)
        print(len(cv_result))
        return max(cv_result['test-auc-mean'])

    # range
    xgbBO = BayesianOptimization(xgb_eval, {'gamma': (0, 5),
                                            'max_depth': (2, 5),
                                            'reg_alpha': (0, 5),
                                            'reg_lambda': (0, 3),
                                            'min_child_weight': (5, 50),
                                            'subsample': (0.5, 1),
                                            'colsample_bytree': (0.1, 1)}, random_state=random_seed)
    # optimize
    xgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return xgbBO.max





def pipe_cv_evaluate(X,y,pipeline_estimator,cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),groups=None,X_test=None,y_test=None):
    '''
    单个Pipeline的CV评估的结果
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    pipeline_estimator:pipeline模型
    cv: 数据集切分方法
    groups: 数据分组
    X_test:测试X数据 可不指定
    y_test:测试y数据 可不指定
    
    return:
    detail_result: 每一折cv的结果 pandas.DataFrame对象
    statistic_result 最终各指标的结果 dict对象
    '''
    valid_auc_list = []
    valid_ks_list = []
    train_auc_list = []
    train_ks_list = []
    if X_test is None:
        test_auc_list = np.nan
        test_ks_list = np.nan
    else:
        test_auc_list = []
        test_ks_list = []
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):

        ##取出当前轮使用的的训练数据和验证数据
        X_valid,y_valid = X.iloc[valid_index],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        ##模型训练
        pipeline_estimator.fit(X,y)
        ##模型预测
        y_pred_train = pipeline_estimator.predict_proba(X_train)[:,1]
        y_pred_valid = pipeline_estimator.predict_proba(X_valid)[:,1]
        ##结果评估
        train_auc = plot_roc_curve(y_train,y_pred_train)
        train_ks = plot_ks_curve(y_pred_train,y_train, is_score=False, n=10)
        valid_auc = plot_roc_curve(y_valid,y_pred_valid)
        valid_ks = plot_ks_curve(y_pred_valid,y_valid, is_score=False, n=10)
        ##结果记录
        train_auc_list.append(train_auc)
        train_ks_list.append(train_ks)
        valid_auc_list.append(valid_auc)
        valid_ks_list.append(valid_ks)
        if X_test is not None:
            y_pred_test = pipeline_estimator.predict_proba(X_test)[:,1]
            test_auc = plot_roc_curve(y_test,y_pred_test)
            test_ks = plot_ks_curve(y_pred_test,y_test, is_score=False, n=10)
            test_auc_list.append(test_auc)
            test_ks_list.append(test_ks)
        
    detail_result = pd.DataFrame(data={'test_auc':test_auc_list,
                                          'test_ks':test_ks_list,
                                          'valid_auc':valid_auc_list,
                                          'valid_ks':valid_ks_list,
                                          'train_ks':train_ks_list,
                                          'train_auc':train_auc_list
                                          })
    statistic_result = { 'train_auc_mean':np.mean(train_auc_list),
                        'train_auc_std':np.std(train_auc_list),
                        'train_ks_mean':np.mean(train_ks_list),
                        'train_ks_std':np.std(train_ks_list),
                        'valid_auc_mean':np.mean(valid_auc_list),
                        'valid_auc_std':np.std(valid_auc_list),
                        'valid_ks_mean':np.mean(valid_ks_list),
                        'valid_ks_std':np.std(valid_ks_list),
                        'test_auc_mean':np.mean(test_auc_list),
                        'test_auc_std':np.std(test_auc_list),
                        'test_ks_mean':np.mean(test_ks_list),
                        'test_ks_std':np.std(test_ks_list)
                   }
    print('train AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(train_auc_list), np.std(train_auc_list)),\
          'train KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(train_ks_list), np.std(train_ks_list)))
    print('valid AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(valid_auc_list), np.std(valid_auc_list)),\
          'valid KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(valid_ks_list), np.std(valid_ks_list)))
    if X_test is not None:
        print('test AUC:{0:.4f}, std:{1:.4f}.'.format(np.mean(test_auc_list), np.std(test_auc_list)),\
              'test KS:{0:.4f}, std:{1:.4f}.'.format(np.mean(test_ks_list), np.std(test_ks_list)))
    return detail_result,statistic_result

def pipe_train_test_evaluate(data_dict,pipeline_estimator):
    '''
    进行pipeline在训练集和多个测试集上面的评估
    data_dict: 多个数据集组成的dict train test_xxx等等key
    pipeline_estimator: pipeline模型
    
    return:
    detail_result: 预测的概率结果和真实标签
    statistic_result: 预测的指标
    '''
    ##取出训练集 进行模型训练
    X_train = data_dict.get('train').get('X')
    y_train = data_dict.get('train').get('y')
    pipeline_estimator.fit(X_train,y_train)
    statistic_result = {}
    detail_result = {}
    ##进行模型预测
    for key in data_dict.keys():
        temp_X = data_dict.get(key).get('X')
        temp_y = data_dict.get(key).get('y')
        temp_predict = pipeline_estimator.predict_proba(temp_X)[:,1]
        statistic_result[key+'_auc'] = plot_roc_curve(temp_y,temp_predict)
        statistic_result[key+'_ks'] = plot_ks_curve(temp_predict, temp_y, is_score=False, n=10)
        detail_result[key] = {'predict':temp_predict,'true':temp_y}
    return detail_result,statistic_result


## 构建pipeline模型
def make_pipeline_model(numeric_feature,category_feature,estimator,X=None,y=None):
    '''
    通过指定类别型和数值型特征构建以及指定的模型构建pipeline,如果给出数据集就完成训练,最终返回pipeline模型
    numeric_feature: 数值特征 list
    category_feature: 类别特征 list
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    
    return:
    pipeline_model
    '''
    feature_def = gen_features(
        columns=category_feature,
        classes=[CategoricalDomain,CategoricalImputer,LabelBinarizer]
        )
    mapper_numerical = DataFrameMapper([
            (numeric_feature,[ContinuousDomain(),SimpleImputer(strategy='mean'),StandardScaler()])
    ])
    mapper_category = DataFrameMapper(feature_def)
    mapper = FeatureUnion([('mapper_numerical',mapper_numerical),('mapper_category',mapper_category)])
    pipeline_model = PMMLPipeline([
        ('mapper',mapper),
        ('classifier',estimator)
    ])
    if X is not None and y is not None:
        pipeline_model.fit(X,y)
    return pipeline_model




