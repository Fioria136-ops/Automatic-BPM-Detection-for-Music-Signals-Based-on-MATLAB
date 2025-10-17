import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def Onsetsdetect_ZCR(x, fs, threshold, frequencyRange):
    """
    基于过零率(ZCR)的起始点检测函数

    参数：
        x              : 输入音频信号
        fs             : 采样率
        threshold      : 阈值系数
        frequencyRange : [low, high] Hz 频率范围

    返回：
        onsetTime   : 起始点时间 (秒)
        onsetValues : 起始点 ZCR 值
        zcr         : ZCR 序列
        t           : 时间轴 (秒)
        Threshold   : 动态阈值
    """
    # ======= 带通滤波 =======
    b, a = butter(2, np.array(frequencyRange)/(fs/2), btype='bandpass')
    x = filtfilt(b, a, x)

    # ======= 参数设置 =======
    windowSize = 2048
    hopSize = 1024
    numFrames = int(np.floor((len(x) - windowSize)/hopSize)) + 1

    zcr = np.zeros(numFrames)
    t = (np.arange(numFrames) * hopSize) / fs

    # 计算每帧 ZCR
    for n in range(numFrames):
        start = n * hopSize
        end = start + windowSize
        if end > len(x):
            break
        frame = x[start:end]
        zcr[n] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

    # 归一化 0~1
    zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)

    # 动态阈值
    Threshold = threshold * np.mean(zcr)

    # 检测局部最大值
    std_zcr = np.std(zcr)
    fws = 2 * std_zcr  # 最小显著性
    idx, _ = find_peaks(zcr, prominence=fws)

    onsetValues = zcr[idx]
    onsetTime = t[idx]

    return onsetTime, onsetValues, zcr, t, Threshold
