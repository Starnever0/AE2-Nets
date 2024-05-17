import math


def next_batch(X1, X2, batch_size):
    # 用来生成大小为 batch_size 的数据批次
    tot = X1.shape[0] # 总样本数
    total = math.ceil(tot / batch_size)  # fix the last batch
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, (i + 1))
