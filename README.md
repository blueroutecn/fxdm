# fxdm
工业优化模型构建的自建代码库

## Requirements
scipy>=1.2.1
pandas>=0.20.3,<=0.23.4
scikit-learn>=0.20.3
sklearn_pandas>=1.8.0
sklearn2pmml>=0.44.0
lightgbm>=2.2.3
xgboost>=0.82
matplotlib>=3.0.3
seaborn>=0.9.0
bayesian-optimization>=1.0.1
tsfresh>=0.11.2
featuretools>=0.7.0
concurrent-utils>=0.2.0
missingno>=0.4.1
xlrd>=1.2.0

## 代码块构成
### fx_utills 基础工具包
+ **timecount**函数 计算运行时间的装饰函数
+ **is_monotonic**函数 判断序列是否单调
+ **check_unique**函数 判断序列是否存在重复值
+ **check_non_intersect**函数 判断序列x和序列y是否存在交集
+ **psi**函数 计算psi
+ **model_y_generator**函数 自由Y相关变量转Y标签
+ **model_result_combine**函数 多模型预测结果dict转换工作
+ **data_sample**函数 数据集划分函数
 
### fx_evaluation 评估工具包
+ **plot_confusion_matrix**函数 绘制混淆矩阵图
+ **plot_ks_curve**函数 绘制KS曲线并返回KS值
+ **plot_roc_curve**函数 绘制ROC曲线并返回AUC值
+ **plot_multi_roc_curve**函数 多模型绘制ROC曲线并返回AUC值
+ **plot_multi_roc_curve_dict_type** 多模型绘制ROC曲线并返回AUC值 使用dict作为数据输入
+ **plot_reject_bad_curve**函数 多模型绘制通过率曲线 
+ **plot_multi_reject_bad_curve_dict_type**函数 多模型绘制通过率曲线 使用dict作为数据输入
+ **plot_PR_curve**函数 绘制PR曲线
+ **plot_multi_PR_curve_dict_type**函数 多模型绘制PR曲线 使用dict作为数据输入
+ **plot_validation_curve**函数 绘制验证曲线
+ **plot_learning_curve**函数 绘制学习曲线
+ **plot_density_curve**函数 绘制0\1预测概率分布图
 
### fx_dataclean 数据清洗包
+ **data_clean**函数 数据清洗函数
 
### fx_cv 数据切分包
+ **TimeGroupSplit**类 数据集时间维度拆分（分组实现方案）
 
### fx_preprocess 数据预处理包
+ **data_simple_imputer**函数 缺失值添补
+ **data_missing_indicator**函数 缺失值标记
+ **SimpleOutlierIndicator**类 分位数异常值标记
 
### fx_model 模型训练包
+ **feature_rank_calculator_lgb**函数 计算特征重要度（LightGbm模型）
+ **feature_rank_split_calculator_lgb**函数 计算特征重要度（多个LightGbm模型加CV）
+ **gbm_cv_evaluate**函数 CV评估（LightGbm模型）
+ **feature_selector**函数 前向stepwise特征筛选（LightGbm模型）
+ **bayes_parameter_opt_lgb**函数 贝叶斯超参数筛选（LightGbm模型）
+ **bayes_parameter_opt_xgb**函数 贝叶斯超参数筛选（Xgboost模型）
+ **make_pipeline_model**函数 通过指定的模型构建pipeline
 
## 工业优化建模流程
参考info下面的pdf文档
 
## 模型训练
+ **daemon** 标准型融合模型训练评估流程 有python文件和notebook文件两个版本
+ **EnsembleSVM** 组合SVM模型训练评估流程 有python文件和notebook文件两个版本
+ **EnsembleGBDT** 参数扰动的组合GDBT模型训练评估流程 有python文件和notebook文件两个版本
 
## 后续优化
+ 其他模型的相关代码
+ pipeline参数筛选的相关代码
