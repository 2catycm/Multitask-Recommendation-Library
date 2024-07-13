# %%
# dataset loading
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearnex import patch_sklearn, config_context
patch_sklearn()
import pandas as pd
# 'os' module provides functions for interacting with the operating system 
import os
# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np
# 'Pandas' is used for data manipulation and analysis
import pandas as pd
# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
from matplotlib import pyplot as plt
# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns
# to suppress warnings
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import tree
from sklearn.tree import _tree
from sklearn.base import is_classifier # 用于判断是回归树还是分类树
from dtreeviz.colors import adjust_colors # 用于分类树颜色（色盲友好模式）
import seaborn as sns #用于回归树颜色
from matplotlib.colors import Normalize # 用于标准化RGB数值
import graphviz # 插入graphviz库
import os
plt.style.use("default")
warnings.filterwarnings("ignore") 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# %%
# https://blog.csdn.net/wtySama/article/details/105316240
import matplotlib
print(matplotlib.matplotlib_fname())
# https://blog.csdn.net/jlb1024/article/details/98037525

# %%
from pathlib import Path
project_root = Path('.').resolve()
print(project_root)

# %%
from munch import DefaultMunch, Munch
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from torch.utils.data import DataLoader
import os
import joblib

memory = joblib.Memory('./joblib_tmp', verbose=1)

dumb_num = 1
categorical_num, numerical_num, labels_num = (16, 63, 2)
# cate_names = [i for i in range(categorical_num)]
# numeric_names = [i for i in range(categorical_num, categorical_num+numerical_num)]
# label_names = [i for i in range(categorical_num+numerical_num, categorical_num+numerical_num+labels_num)]
print(categorical_num, numerical_num, labels_num)
@memory.cache
def get_ds():
    train_data = TabularDataset((project_root/'data/AliExpress_NL/train.csv').as_posix())
    test_data = TabularDataset((project_root/'data/AliExpress_NL/test.csv').as_posix())
    return train_data, test_data
@memory.cache
def get_processed():
    train_data, test_data = get_ds()
    train_data.drop(columns=['search_id'], inplace=True)
    test_data.drop(columns=['search_id'], inplace=True)
    cate_names = train_data.columns[:categorical_num]
    numeric_names = train_data.columns[categorical_num:categorical_num+numerical_num]
    label_names = train_data.columns[categorical_num+numerical_num:categorical_num+numerical_num+labels_num]
    def get_cated():
            return train_data[cate_names].astype('category'), test_data[cate_names].astype('category')
    train_data[cate_names], test_data[cate_names] = get_cated()
    return train_data, test_data
train_data, test_data = get_processed()
print(train_data.info())

# %%
# train_data.head()
test_data.head()

# %%
cate_names = train_data.columns[:categorical_num]
numeric_names = train_data.columns[categorical_num:categorical_num+numerical_num]
label_names = train_data.columns[categorical_num+numerical_num:categorical_num+numerical_num+labels_num]
cate_names, numeric_names, label_names

# %%
y_learn = 0
y_name = label_names[y_learn]

# %% [markdown]
# ## 基本分析

# %%
# import autogluon.eda.auto as auto
# auto.covariate_shift_detection(train_data=train_data, test_data=test_data, label=y_name)
# plt.savefig(f'runs/analysis_fig/covariate_shift_detection_for {y_name}.png')

# %%
from sklearn.model_selection import train_test_split
@memory.cache
def get_single_label():
    return train_data[label_names[0]]+train_data[label_names[1]], test_data[label_names[0]]+test_data[label_names[1]]
@memory.cache
def get_sampled():
    tr_y, te_y = get_single_label()
    _, train_data_sampled = train_test_split(train_data, test_size=0.001, stratify=tr_y, random_state=42)
    _, test_data_sampled = train_test_split(test_data, test_size=0.001, stratify=te_y, random_state=42)
    return train_data_sampled, test_data_sampled
train_data_sampled, test_data_sampled = get_sampled()

# %%
len(test_data)/(len(train_data)+len(test_data))

# %%
from sklearn.model_selection import train_test_split
@memory.cache
def get_concat():
    return pd.concat([train_data, test_data], ignore_index=True)
@memory.cache
def get_concated_label():
    tr_y, te_y = get_single_label()
    return pd.concat([tr_y, te_y], ignore_index=True)
@memory.cache
def get_reshuffled():
    cated = get_concat()
    y = get_concated_label()
    return train_test_split(cated, test_size=0.3, stratify=y, random_state=42)
train_data, test_data = get_reshuffled()

# %%
@memory.cache
def get_single_label_again():
    return train_data[label_names[0]]+train_data[label_names[1]], test_data[label_names[0]]+test_data[label_names[1]]
@memory.cache
def get_sampled_again():
    tr_y, te_y = get_single_label_again()
    _, train_data_sampled = train_test_split(train_data, test_size=0.001, stratify=tr_y, random_state=42)
    _, test_data_sampled = train_test_split(test_data, test_size=0.001, stratify=te_y, random_state=42)
    return train_data_sampled, test_data_sampled
train_data_sampled, test_data_sampled = get_sampled_again()

# %%
len(train_data_sampled), len(test_data_sampled)

# %%
# auto.covariate_shift_detection(train_data=train_data_sampled, test_data=test_data_sampled, label=y_name)
# auto.covariate_shift_detection(train_data=train_data_sampled, test_data=test_data_sampled, label=label_names[1])
# 用来筛选是否有 id 类的 特征存在，使得 测试集和训练集中很不一样，就是id单调递增而已。

# %%
# auto.dataset_overview(train_data=train_data, test_data=test_data, label=y_name)

# # %%
# import autogluon.eda.auto as auto
# for i in range(2):
#     y_name = label_names[i]
#     other_name = label_names[1-i]
#     auto.target_analysis(train_data=train_data[list(numeric_names)+[y_name]], label=y_name)
#     state = auto.quick_fit(
#         train_data[list(numeric_names)+[y_name]], 
#         y_name, 
#         return_state=True,
#         show_feature_importance_barplots=True
#     )

# # %%
# state = auto.partial_dependence_plots(train_data_sampled[list(numeric_names)+[y_name]], label=y_name, return_state=True)

# # %%
# # auto.analyze_interaction(train_data=XY, x=y_name, y='Age', hue='Gender')
# # 41, 45, 31, 28
# # auto.analyze_interaction(train_data=XY, x=y_name, y='Weight', hue='Gender') # 女人本身就比男人轻，没什么好说的。 # 患病的人确实轻一点
# auto.analyze_interaction(train_data=train_data, x=numeric_names[28-1], y=numeric_names[45-1], hue=y_name) 
# # auto.analyze_interaction(train_data=XY, x=y_name, y='FT4', hue='Gender')

# # %%
# auto.analyze_interaction(x=numeric_names[45-1], hue=y_name, train_data=train_data)

# # %%
# # This parameter specifies how many standard deviations above mean anomaly score are considered
# # to be anomalies (only needed for visualization, does not affect scores calculation).
# threshold_stds = 3
# auto.detect_anomalies(
#     train_data=train_data,
#     test_data=test_data,
#     label=y_name,
#     threshold_stds=threshold_stds,
#     show_top_n_anomalies=None,
#     fig_args={
#         'figsize': (6, 4)
#     },
#     chart_args={
#         'normal.color': 'lightgrey',
#         'anomaly.color': 'orange',
#     }
# )

# %% [markdown]
# ## interpretable的决策树分析

# %%
# from draw_tree import classification_and_draw, regression_and_draw
# # x = math_X+answer_features+math_Y[:1]
# # y = math_Y[-1]
# y_learn = 0
# y_name = label_names[y_learn]
# X_columns = list(numeric_names)+list(cate_names)
# dt = classification_and_draw(train_data[X_columns], train_data[y_name], 
#                              X_columns, train_data, 
#                         class_names=[f"not {y_name}", y_name], 
#                         path=f"runs/analysis_fig/{y_name}", 
#                         dummy_indicator="()", replacement={})


# # %%
# y_test = test_data[y_name].values
# y_pred_prob = dt.predict_proba(test_data[X_columns])[:, 1]

# # %%
# import draw_metrics
# from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
# # draw_metrics.fast_evaluation(y_test, y_pred_prob, threshold=0.5)
# # {'roc_auc': 0.75364350774827,
# #  'accuracy': 0.635513820000493,
# #  'balanced_accuracy': 0.6865827700885063,
# #  'mcc': 0.11183493419822582,
# #  'f1': 0.08031662178944289,
# #  'precision': 0.042462848207854266,
# #  'recall': 0.7399473421796139}
# # draw_metrics.fast_evaluation(y_test, y_pred_prob, threshold=None, metric=f1_score)
# # draw_metrics.fast_evaluation(y_test, y_pred_prob, threshold=None, metric=draw_metrics.f1_score)
# draw_metrics.fast_evaluation(y_test[:10000], y_pred_prob[:10000], threshold=None, metric=draw_metrics.f1_score)

# # %%
# # draw_metrics.roc_auc_score(y_test, y_pred_prob)
# %timeit draw_metrics.roc_auc_score(y_test, y_pred_prob)

# # %%
# %timeit roc_auc_score(y_test, y_pred_prob) # 加速30倍

# # %%
# from PIL import Image
# from IPython.display import SVG
# image =SVG(f'runs/analysis_fig/{y_name}.svg')
# image

# %% [markdown]
# ## 自动机器学习分析

# %%
for y_learn in range(2):
    y_name = label_names[y_learn]
    predictor = TabularPredictor(label=y_name, verbosity=2, 
                                eval_metric='roc_auc',
        # sample_weight = 'balance_weight',
        sample_weight = 'auto_weight',
                                
                                ).fit(
    # predictor = TabularPredictor(label=y_name, verbosity=2, eval_metric='accuracy').fit(
        train_data, 
        tuning_data=test_data, 
        # presets='best_quality', time_limit=60*60*2, 
        # num_gpus=4
        )
    # predictor = TabularPredictor(label=y_name, verbosity=0).fit(train_data, tuning_data=test_data)
    res = predictor.evaluate(test_data, silent=True)
    print(res)

    # %%
    # predictor = TabularPredictor.load("AutogluonModels/ag-20230729_111437/")
    # dir(predictor)
    # predictor.fit_summary()
    # predictor.evaluate(test_data, silent=True)
    fi = predictor.feature_importance(test_data, silent=True, num_shuffle_sets=None,  include_confidence_band=True)
    # predictor.leaderboard(test_data, silent=True, extra_info=True, extra_metrics=['accuracy', 'balanced_accuracy', 'mcc', 'f1', 'precision', 'recall'])

    # %%
    fi.to_csv(f'runs/analysis_fig/feature_importance_for {y_name}.csv', index=True)
    plt.savefig(f"runs/analysis_fig/feature_importance_for {y_name}.png")

# %%
    plt.figure(figsize=(20, 10))
    plt.bar(fi.index.head(10), fi['importance'].head(10))


    y_test = test_data[y_name].values
    y_test
    y_pred_prob = predictor.predict_proba(test_data, as_multiclass=False)
    y_pred_prob[:10]

# %%
    # from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
    # import draw_metrics
    # best_threshold_value, best_score = draw_metrics.best_threshold(y_test, y_pred_prob, metric=balanced_accuracy_score)
    # # best_threshold_value, best_score = best_threshold(y_test, y_pred_prob, metric=matthews_corrcoef)
    # # best_threshold_value, best_score = best_threshold(y_test, y_pred_prob)
    # best_threshold_value, best_score
    # draw_metrics.fast_evaluation(y_test, y_pred_prob, threshold=best_threshold_value)
    
    best_threshold_value = 0.5

# %%
    import scikitplot as skplt
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred_prob>best_threshold_value, normalize=False)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred_prob>best_threshold_value, normalize=True)
    plt.savefig(f"runs/analysis_fig/confusion_matrix_for {y_name}.png")

# %%
    # sns.displot(y_pred_prob)
    def draw_probs(y_pred_prob, y_test):
        sns.scatterplot(y_pred_prob[y_test==0], label=f"not {y_name}")
        sns.scatterplot(y_pred_prob[y_test==1], label=f"{y_name}", color="red")
    draw_probs(y_pred_prob, y_test)
    plt.savefig(f"runs/analysis_fig/prob_separate_test_for {y_name}.png")

# %%
    y_pred_train_prob = predictor.predict_proba(train_data, as_multiclass=False)
    y_train = train_data[y_name].values
    draw_probs(y_pred_train_prob, y_train)
    plt.savefig(f"runs/analysis_fig/prob_separate_train_for {y_name}.png")

# %%
# 自动画图 工程
# https://zhuanlan.zhihu.com/p/372676897
# https://blog.csdn.net/wuli_xin/article/details/106612952


# 高端论文画图、动画
# https://zhuanlan.zhihu.com/p/144973329


# %%
    import draw_metrics
    draw_metrics.plot_auc(y_test, y_pred_prob, xlabel="以为点击/实际没点", ylabel="实际找出/所有点击")
    plt.savefig(f"runs/analysis_fig/roc_auc_plot {y_name}.png")

    # https://zhuanlan.zhihu.com/p/405658103
    draw_metrics.plot_pr(y_test, y_pred_prob, xlabel="实际找出/所有点击", ylabel="确实点击/认为点击")
    plt.savefig(f"runs/analysis_fig/pr_auc_plot {y_name}.png")

# %%


# %%
# scikit-plot 对结果进行可视化评估
    import scikitplot as skplt
    def get_skprob(binary_prob):
        return np.array([1 - binary_prob, binary_prob]).T
    # skplt.metrics.plot_roc(y_test, get_skprob(y_pred_prob))

# %%
# skplt.metrics.plot_precision_recall(y_test, get_skprob(y_pred_prob), cmap='nipy_spectral')

# %%
    predictor.get_model_best()

    # %%
    skplt.metrics.plot_calibration_curve(y_test,
                                        probas_list=[get_skprob(y_pred_prob)],
                                        clf_names=[predictor.get_model_best()],
                                        n_bins=10)
    plt.savefig(f"runs/analysis_fig/calibration_curve {y_name}.png")

# %%
    skplt.metrics.plot_cumulative_gain(y_true=y_test, y_probas=get_skprob(y_pred_prob))
    plt.savefig(f"runs/analysis_fig/cumulative_gain {y_name}.png")


    # %%
    skplt.metrics.plot_ks_statistic(y_true=y_test, y_probas=get_skprob(y_pred_prob))
    plt.savefig(f"runs/analysis_fig/ks_statistic {y_name}.png")


    # %%
    skplt.metrics.plot_lift_curve(y_true=y_test, y_probas=get_skprob(y_pred_prob))
    plt.savefig(f"runs/analysis_fig/lift_curve {y_name}.png")


    # %%
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    # https://zhuanlan.zhihu.com/p/555215223
    # pca = make_pipeline(StandardScaler(), SimpleImputer(), PCA(random_state=1))
    pca = make_pipeline(StandardScaler(), PCA(random_state=1))
    pca.fit(train_data[numeric_names])
    skplt.decomposition.plot_pca_2d_projection(pca, train_data[numeric_names], train_data[y_name])
    plt.savefig(f"runs/analysis_fig/pca_2d_projection {y_name}.png")

    # %%
    # dir(pca)
    pca.steps

    # %%
    skplt.decomposition.plot_pca_component_variance(pca.steps[1][1])
    plt.savefig(f"runs/analysis_fig/pca_component_variance {y_name}.png")

# %%


# %%



