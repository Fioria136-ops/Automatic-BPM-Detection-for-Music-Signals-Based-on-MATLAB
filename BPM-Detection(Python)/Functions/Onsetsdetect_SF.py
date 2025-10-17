import numpy as np
from scipy.signal import butter, filtfilt, stft
from scipy.signal.windows import hamming
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks

def Onsetsdetect_SF(x, fs, threshold, frequencyRange):
    """
    基于谱流(SF)的起始点检测函数

    参数：
        x              : 输入音频信号
        fs             : 采样率
        threshold      : 阈值系数
        frequencyRange : [low, high] Hz 频率范围

    返回：
        onsetTime   : 起始点时间 (秒)
        onsetValues : 起始点 SF 值
        sf          : 谱流序列
        t           : 时间轴 (秒)
        Threshold   : 动态阈值
    """
    # ======= 带通滤波 =======
    b, a = butter(2, np.array(frequencyRange)/(fs/2), btype='bandpass')
    x = filtfilt(b, a, x)

    # ======= STFT 参数 =======
    windowSize = 2048
    hopSize = 1024
    f, t, Zxx = stft(x, fs=fs, window=hamming(windowSize, sym=False),
                     nperseg=windowSize, noverlap=windowSize-hopSize, nfft=windowSize)
    X = np.abs(Zxx)

    # ======= 计算谱流 SF =======
    sf = np.zeros(X.shape[1])
    for n in range(1, X.shape[1]):
        sf[n] = np.sum(np.maximum(X[:, n] - X[:, n-1], 0))
    
    # 归一化 0~1
    sf = minmax_scale(sf)

    # 动态阈值
    Threshold = threshold * np.mean(sf)

    # 检测局部最大值
    std_sf = np.std(sf)
    fws = 100 * std_sf  # 这里用 MinDistance/MinProminence 替代 MATLAB MinSeparation
    idx, _ = find_peaks(sf, distance=fws)
    
    # 筛选超过阈值的起始点
    valid_idx = sf[idx] > Threshold
    onsetTime = t[idx[valid_idx]]
    onsetValues = sf[idx[valid_idx]]

    return onsetTime, onsetValues, sf, t, Threshold
