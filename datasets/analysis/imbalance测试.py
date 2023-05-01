# %%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent.parent
import sys
sys.path.append(project_directory.as_posix())

# %%
import numpy as np
# import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn()

# %%
from munch import DefaultMunch, Munch
params = DefaultMunch()
params.dataset_name = 'AliExpress_NL'
params.dataset_path = (project_directory/'data').as_posix()

# %%
from datasets import get_dataset
import os
from torch.utils.data import DataLoader
train_dataset = get_dataset(params.dataset_name, os.path.join(
        params.dataset_path, params.dataset_name) + '/train.csv')
test_dataset = get_dataset(params.dataset_name, os.path.join(
    params.dataset_path, params.dataset_name) + '/test.csv')


# %%
import joblib
memory = joblib.Memory('./joblib_tmp', verbose=1)
@memory.cache
def get_multi_numeric():
    # X_train = np.hstack((train_dataset.categorical_data, train_dataset.numerical_data))
    X_train = train_dataset.numerical_data
    # X_train = train_dataset.categorical_data
    # y_train = np.sum(train_dataset.labels, axis=1)
    y_train = train_dataset.labels
    # X_test = np.hstack((test_dataset.categorical_data, test_dataset.numerical_data))
    X_test = test_dataset.numerical_data
    # X_test = test_dataset.categorical_data
    # y_test = np.sum(test_dataset.labels, axis=1)
    y_test = test_dataset.labels
    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = get_multi_numeric()

# %%
# from sklearn.linear_model import LogisticRegression
# clf0 = LogisticRegression(random_state=0)
# clf0.fit(X_train, y_train[:, 0])
# clf1 = LogisticRegression(random_state=0)
# clf1.fit(X_train, y_train[:, 1])

from imblearn.ensemble import BalancedBaggingClassifier
clf0 = BalancedBaggingClassifier(random_state=0, verbose=1, n_jobs=128)
clf0.fit(X_train, y_train[:, 0])
clf1 = BalancedBaggingClassifier(random_state=0, verbose=1, n_jobs=128)
clf1.fit(X_train, y_train[:, 1])

# %%
from matplotlib import pyplot as plt
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.inspection import DecisionBoundaryDisplay

def plot_contour(clf, X, y, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    plt.savefig(f'fig/{filename}.png')

def plot_auc(y_test, y_pred, filename):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve for {filename}')
    plt.legend(loc="lower right")
    plt.savefig(f'fig/{filename}.png')
    return roc_auc

def evaluate_and_draw(model, X_test, y_test, model_name='tree', classes = ['impression', 'click']):
    print(f"模型自带分数：{model.score(X_test, y_test)}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    print(f"F1: {f1_score(y_test, y_pred, average='macro')}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f'log_loss: {log_loss(y_test, y_proba)}')
    # 多分类，取第一类的auc
    auc = plot_auc(y_test, y_proba[:, 1], f'{model_name}_auc') 
    print(f"auc：{auc}")
    y_pred = model.predict(X_test)
    # classes = ['impression', 'click', 'purchase']
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.savefig(f'fig/{model_name}.png')


# %%
evaluate_and_draw(clf0, X_test, y_test[:, 0], 'balanced_bag_two0')
evaluate_and_draw(clf1, X_test, y_test[:, 1], 'balanced_bag_two1')
#%%
plot_contour(clf, X_test[:, :2], y_test[:, 0], 'logistic_two0_contour')


# %%
