import numpy as np

def moving_average(a, window_size):
    """
    计算一维数组的移动平均值，支持任意 window_size。

    参数：
    a : list 或 np.array
        输入序列
    window_size : int
        窗口大小

    返回：
    np.array
        平滑后的序列
    """
    a = np.array(a, dtype=float)
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")
    if window_size > len(a):
        raise ValueError("window_size 不能大于输入序列长度")

    # 核心部分：卷积实现
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(a, weights, mode='valid')

    # 为了保持长度一致，前后补值（复制边界值）
    pad_left = np.full((window_size // 2,), smoothed[0])
    pad_right = np.full((len(a) - len(smoothed) - len(pad_left),), smoothed[-1])

    return np.concatenate([pad_left, smoothed, pad_right])