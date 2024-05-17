from utils.Dataset import Dataset
from model import model
from utils.print_result import print_result
import os
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
if __name__ == '__main__':
    data = Dataset('handwritten_2views')

    task_list = ['clustering', 'classification']
    task = task_list[1]

    x1, x2, gt = data.load_data()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    n_clusters = len(set(gt))

    act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    # 中间隐藏层200个节点
    dims_ae1 = [240, 200]
    dims_ae2 = [216, 200]
    dims_dg1 = [64, 200]
    dims_dg2 = [64, 200]

    para_lambda = 1
    batch_size = 100
    lr_pre = 1.0e-3
    lr_ae = 1.0e-3
    lr_dg = 1.0e-3
    lr_h = 1.0e-1
    epochs_pre = 10
    epochs_total = 20
    act = [act_ae1, act_ae2, act_dg1, act_dg2]
    dims = [dims_ae1, dims_ae2, dims_dg1, dims_dg2]
    lr = [lr_pre, lr_ae, lr_dg, lr_h]
    epochs_h = 50
    epochs = [epochs_pre, epochs_total, epochs_h]

    H, gt = model(x1, x2, gt, para_lambda, dims, act, lr, epochs, batch_size)
    
    # import scipy.io as sio
    # # 加载.mat文件
    # data = sio.loadmat('H.mat')
    # # 从字典中提取数据
    # H = data['H']
    # gt = data['gt'].ravel()

    if task == 'clustering':
        print_result(n_clusters, H, gt, task)
    elif task == 'classification':
        print_result(0, H, gt, task)
