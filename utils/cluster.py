from sklearn.cluster import KMeans, spectral_clustering
from . import metrics
import numpy as np


def cluster(n_clusters, features, labels, count=10):
    # 使用学习到的h进行聚类任务
    """
    :param n_clusters: number of categories
    :param features: input to be clustered
    :param labels: ground truth of input
    :param count:  times of clustering
    :return: average acc and its standard deviation,
             average nmi and its standard deviation
    """
    pred_all = []
    for i in range(count):
        km = KMeans(n_clusters=n_clusters)
        pred = km.fit_predict(features)
        pred_all.append(pred)
    gt = np.reshape(labels, np.shape(pred))
    if np.min(gt) == 1:
        gt -= 1
    acc_avg, acc_std = get_avg_acc(gt, pred_all, count)
    nmi_avg, nmi_std = get_avg_nmi(gt, pred_all, count)
    ri_avg, ri_std = get_avg_RI(gt, pred_all, count)
    f1_avg, f1_std = get_avg_f1(gt, pred_all, count)
    return acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std


def get_avg_acc(y_true, y_pred, count):
    acc_array = np.zeros(count)
    for i in range(count):
        acc_array[i] = metrics.acc(y_true, y_pred[i])
    acc_avg = acc_array.mean()
    acc_std = acc_array.std()
    return acc_avg, acc_std


def get_avg_nmi(y_true, y_pred, count):
    nmi_array = np.zeros(count)
    for i in range(count):
        nmi_array[i] = metrics.nmi(y_true, y_pred[i])
    nmi_avg = nmi_array.mean()
    nmi_std = nmi_array.std()
    return nmi_avg, nmi_std


def get_avg_RI(y_true, y_pred, count):
    # 计算兰德指数，用来评估聚类效果：RI = (TP + TN) / (TP + FP + FN + TN)
    # TP是真正例，FP是假正例，FN是假负例，TN是真负例
    # RI的取值范围是[0, 1]，值越大表示聚类效果越好
    RI_array = np.zeros(count)
    for i in range(count):
        RI_array[i] = metrics.rand_index_score(y_true, y_pred[i])
    RI_avg = RI_array.mean()
    RI_std = RI_array.std()
    return RI_avg, RI_std


def get_avg_f1(y_true, y_pred, count):
    # 计算F1值，用来评估聚类效果：F1 = 2 * (precision * recall) / (precision + recall)
    # precision是精确率，recall是召回率
    # F1的取值范围是[0, 1]，值越大表示聚类效果越好
    f1_array = np.zeros(count)
    for i in range(count):
        f1_array[i] = metrics.f_score(y_true, y_pred[i])
    f1_avg = f1_array.mean()
    f1_std = f1_array.std()
    return f1_avg, f1_std


def get_acc(y_true, y_pred):
    if np.min(y_true) == 1:
        y_true -= 1
    acc_array = metrics.acc(y_true, y_pred)
    return acc_array


def get_nmi(y_true, y_pred):
    # 计算归一化互信息，用来评估聚类效果：NMI = 2 * I(y_true, y_pred) / (H(y_true) + H(y_pred))
    # I(y_true, y_pred)是互信息，H(y_true)和H(y_pred)分别是y_true和y_pred的熵
    # NMI的取值范围是[0, 1]，值越大表示聚类效果越好
    if np.min(y_true) == 1:
        y_true -= 1
    acc_array = metrics.nmi(y_true, y_pred)
    return acc_array
