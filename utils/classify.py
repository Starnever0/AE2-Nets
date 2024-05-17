from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def classify(features, labels, count=10):
    """
    使用学习到的h进行分类任务

    :param features: 输入特征
    :param labels: 输入的真实标签
    :param test_size: 测试集比例
    :param count: 分类次数
    :return: 平均精度及其标准差，平均NMI及其标准差，平均RI及其标准差，平均F1值及其标准差
    """
    acc_all = []
    nmi_all = []
    ri_all = []
    f1_all = []

    test_size = 0.2
    
    for i in range(count):
        # 拆分数据集
        h_train, h_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=i)

        # 数据标准化
        scaler = StandardScaler()
        h_train = scaler.fit_transform(h_train)
        h_test = scaler.transform(h_test)

        # 训练逻辑回归分类器
        clf = LogisticRegression(max_iter=1000)
        clf.fit(h_train, y_train)
        y_pred = clf.predict(h_test)

        # 评估性能
        acc_all.append(accuracy_score(y_test, y_pred))
        nmi_all.append(normalized_mutual_info_score(y_test, y_pred))
        ri_all.append(adjusted_rand_score(y_test, y_pred))
        f1_all.append(f1_score(y_test, y_pred, average='weighted'))

    # 计算平均值和标准差
    acc_avg, acc_std = np.mean(acc_all), np.std(acc_all)
    nmi_avg, nmi_std = np.mean(nmi_all), np.std(nmi_all)
    ri_avg, ri_std = np.mean(ri_all), np.std(ri_all)
    f1_avg, f1_std = np.mean(f1_all), np.std(f1_all)

    return acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std
