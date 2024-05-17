from utils.cluster import cluster
from utils.classify import classify
import warnings

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, task,  count=10):
    if task == 'clustering':
        acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std = cluster(n_clusters, H, gt, count=count)
        print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_avg, nmi_avg))
    elif task == 'classification':
        acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std = classify(H, gt, count=count)
        print('classification h   : acc = {:.4f}, f1 = {:.4f}'.format(acc_avg, f1_avg))
