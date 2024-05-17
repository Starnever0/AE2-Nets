import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae
from utils.Net_dg import Net_dg
from utils.next_batch import next_batch
import math
from sklearn.utils import shuffle
import timeit


def model(X1, X2, gt, para_lambda, dims, act, lr, epochs, batch_size):
    """
    Building model
    :rtype: object
    :param X1: data of view1 视图1的数据
    :param X2: data of view2 视图2的数据
    :param gt: ground truth 标签
    :param para_lambda: trade-off factor in objective 权衡因子
    :param dims: dimensionality of each layer 每一层的维度
    :param act: activation function of each net 每个网络的激活函数
    :param lr: learning rate 学习率
    :param epochs: learning epoch 学习周期
    :param batch_size: batch size 批处理大小
    """
    start = timeit.default_timer() #计时器
    err_pre = list() #记录重构误差
    err_total = list() #记录总误差

    # define each net architecture and variable(refer to framework-simplified)
    net_ae1 = Net_ae(1, dims[0], para_lambda, act[0]) #视图1的内ae
    net_ae2 = Net_ae(2, dims[1], para_lambda, act[1]) #视图2的内ae
    net_dg1 = Net_dg(1, dims[2], act[2]) #视图1的外ae
    net_dg2 = Net_dg(2, dims[3], act[3]) #视图2的外ae

    H = np.random.uniform(0, 1, [X1.shape[0], dims[2][0]])
    x1_input = tf.placeholder(np.float32, [None, dims[0][0]])
    x2_input = tf.placeholder(np.float32, [None, dims[1][0]])

    with tf.variable_scope("H"):
        h_input = tf.Variable(xavier_init(batch_size, dims[2][0]), name='LatentSpaceData')
        h_list = tf.trainable_variables()

    # 每个视图的潜在表示h
    fea1_latent = tf.placeholder(np.float32, [None, dims[0][-1]])# 视图1的内ae的输出，即z_half
    fea2_latent = tf.placeholder(np.float32, [None, dims[1][-1]])

    # 公式（3）重建损失，内ae，X与编码再解码后重构的Z
    # 使用重建损失预训练内ae
    loss_pre = net_ae1.loss_reconstruct(x1_input) + net_ae2.loss_reconstruct(x2_input) #两个视图相加
    pre_train = tf.train.AdamOptimizer(lr[0]).minimize(loss_pre)

    # 公式（5）总损失，内ae与外ae
    # 使用总损失训练内ae
    loss_ae = net_ae1.loss_total(x1_input, fea1_latent) + net_ae2.loss_total(x2_input, fea2_latent)
    update_ae = tf.train.AdamOptimizer(lr[1]).minimize(loss_ae, var_list=net_ae1.netpara.extend(net_ae2.netpara))
    z_half1 = net_ae1.get_z_half(x1_input)
    z_half2 = net_ae2.get_z_half(x2_input)

    # 公式（4）退化网络损失，外ae，z_half与h
    # 使用退化网络训练外ae
    loss_dg = para_lambda * (
                net_dg1.loss_degradation(h_input, fea1_latent) + net_dg2.loss_degradation(h_input, fea2_latent))
    update_dg = tf.train.AdamOptimizer(lr[2]).minimize(loss_dg, var_list=net_dg1.netpara.extend(net_dg2.netpara))

    # 使用外ae训练H
    update_h = tf.train.AdamOptimizer(lr[3]).minimize(loss_dg, var_list=h_list)
    g1 = net_dg1.get_g(h_input)# 使用h生成视图1的z_half（g）
    g2 = net_dg2.get_g(h_input)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # init inner AEs 初始化内ae
    for k in range(epochs[0]):
        X1, X2, gt = shuffle(X1, X2, gt)
        for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
            _, val_pre = sess.run([pre_train, loss_pre], feed_dict={x1_input: batch_x1, x2_input: batch_x2})
            err_pre.append(val_pre) #记录预训练重构误差
            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.4f} ".format((k + 1), batch_No,
                                                                                                     val_pre)
            print(output)

    # the whole training process(ADM) 
    num_samples = X1.shape[0] # 样本数
    num_batchs = math.ceil(num_samples / batch_size)  # fix the last batch
    for j in range(epochs[1]):
        X1, X2, H, gt = shuffle(X1, X2, H, gt)
        for num_batch_i in range(int(num_batchs) - 1):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx) # fix the last batch
            batch_x1 = X1[start_idx: end_idx, ...]
            batch_x2 = X2[start_idx: end_idx, ...]
            batch_h = H[start_idx: end_idx, ...]

            batch_g1 = sess.run(g1, feed_dict={h_input: batch_h})
            batch_g2 = sess.run(g2, feed_dict={h_input: batch_h})

            # ADM-step1: optimize inner AEs and
            # 使用总损失训练内ae
            _, val_ae = sess.run([update_ae, loss_ae], feed_dict={x1_input: batch_x1, x2_input: batch_x2,
                                                                  fea1_latent: batch_g1, fea2_latent: batch_g2})

            # get inter - layer features(i.e., z_half)
            batch_z_half1 = sess.run(z_half1, feed_dict={x1_input: batch_x1})
            batch_z_half2 = sess.run(z_half2, feed_dict={x2_input: batch_x2})

            sess.run(tf.assign(h_input, batch_h)) # 更新h_input为batch_h

            # ADM-step2: optimize dg nets
            # 使用退化网络损失训练外ae（dg网络）
            _, val_dg = sess.run([update_dg, loss_dg], feed_dict={fea1_latent: batch_z_half1,
                                                                  fea2_latent: batch_z_half2})

            # ADM-step3: update H
            # 使用外ae训练H
            for k in range(epochs[2]):
                sess.run(update_h, feed_dict={fea1_latent: batch_z_half1, fea2_latent: batch_z_half2})

            batch_h_new = sess.run(h_input) # 如果没有更新h_input，需要加feed_dict：{h_input: batch_h}
            H[start_idx: end_idx, ...] = batch_h_new

            # get latest feature_g for next iteration
            sess.run(tf.assign(h_input, batch_h_new)) # 更新h_input为batch_h_new
            batch_g1_new = sess.run(g1, feed_dict={h_input: batch_h})
            batch_g2_new = sess.run(g2, feed_dict={h_input: batch_h})

            # 计算总损失
            val_total = sess.run(loss_ae, feed_dict={x1_input: batch_x1, x2_input: batch_x2,
                                                     fea1_latent: batch_g1_new, fea2_latent: batch_g2_new})
            err_total.append(val_total)
            output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1),
                                                                                                  (num_batch_i + 1),
                                                                                                  val_total)
            print(output)

    elapsed = (timeit.default_timer() - start)
    print("Time used: ", elapsed)
    # 保存潜在表示h，标签gt，总误差，时间
    scio.savemat('H.mat', mdict={'H': H, 'gt': gt, 'loss_total': err_total, 'time': elapsed,
                                    'x1': X1, 'x2': X2})
    return H, gt #输出潜在表示h，同时返回标签gt


def xavier_init(fan_in, fan_out, constant=1):
    # 初始化神经网络权重
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)