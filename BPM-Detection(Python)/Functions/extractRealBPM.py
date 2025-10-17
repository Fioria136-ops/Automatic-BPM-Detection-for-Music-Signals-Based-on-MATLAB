import re
import numpy as np

def extractRealBPM(fileName):
    """
    从文件名中提取真实 BPM（节拍数）

    参数：
        fileName : str
            音频文件名，例如 'songname_128bpm.wav'

    返回：
        realBPM1, realBPM2, realBPM3, realBPM4, realBPM5 : float
            五个不同倍数的 BPM 值（分别为 1/2, 1/1.5, 1, 1.5, 2 倍）
    """
    # 使用正则表达式匹配 “数字 + bpm” 模式
    match = re.search(r'(\d+)bpm', fileName, re.IGNORECASE)

    if match:
        realBPM = float(match.group(1))  # 提取数字部分并转为浮点数
    else:
        realBPM = np.nan  # 若未匹配到，返回 NaN

    # 生成 5 种倍数形式的 BPM
    realBPM1 = realBPM / 2
    realBPM2 = realBPM / 1.5
    realBPM3 = realBPM
    realBPM4 = realBPM * 1.5
    realBPM5 = realBPM * 2

    return realBPM1, realBPM2, realBPM3, realBPM4, realBPM5
