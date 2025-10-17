import numpy as np

def dynamicRangeCompression(x, compressionThreshold, compressionRatio):
    """
    动态范围压缩函数

    参数：
        x                   : 输入音频信号（1D numpy array）
        compressionThreshold: 压缩阈值
        compressionRatio    : 压缩比

    返回：
        x_out : 压缩后的信号
    """
    x_out = np.copy(x)
    x_abs = np.abs(x)

    # 找出超过阈值的部分
    aboveThreshold = x_abs > compressionThreshold

    # 对超过阈值的部分进行压缩
    x_out[aboveThreshold] = np.sign(x[aboveThreshold]) * (
        compressionThreshold + (x_abs[aboveThreshold] - compressionThreshold) / compressionRatio
    )

    return x_out
