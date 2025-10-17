import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from dynamicRangeCompression import dynamicRangeCompression
from Onsetsdetect_SF import Onsetsdetect_SF
from Onsetsdetect_ZCR import Onsetsdetect_ZCR


def analyzeBPM_v12(
    audioFilePath,
    sampleDuration,
    threshold_SF,
    compressionThreshold,
    compressionRatio,
    compressionStatus,
    ploteachsample,
    Std_factor,
    threshold_ZCR,
    frequencyRange_SF,
    frequencyRange_ZCR
):
    """
    核心 BPM 分析函数：使用 SF（谱流）与 ZCR（过零率）两种方法检测音频节拍，
    通过标准差滤波计算三个可能的 BPM 值并加权输出。

    参数说明：
        audioFilePath          : 音频文件路径
        sampleDuration          : 每次分析的持续时间（秒）
        threshold_SF            : 谱流阈值
        compressionThreshold    : 动态压缩阈值
        compressionRatio        : 压缩比
        compressionStatus       : 是否启用压缩
        ploteachsample          : 是否绘制每个样本的检测图
        Std_factor              : 标准差筛选因子
        threshold_ZCR           : 过零率阈值
        frequencyRange_SF       : SF 分析频率范围 [low, high]
        frequencyRange_ZCR      : ZCR 分析频率范围 [low, high]
    """

    # ========= 读取与预处理音频 =========
    fs, x = wavfile.read(audioFilePath)
    x = x.astype(float)

    # 仅保留单声道
    if x.ndim > 1:
        x = x[:, 0]

    # 归一化（使 RMS = 0.3）
    x_current = np.sqrt(np.mean(x ** 2))
    target_rms = 0.3
    x_gain = target_rms / x_current
    x = x * x_gain

    # 提取平均能量最高的时间片段
    window_size = int(round(sampleDuration * fs))
    step_size = int(round(4 * fs))
    windows = int(np.floor((len(x) - window_size) / step_size) + 1)

    if window_size < len(x):  # 当 sampleDuration < 音频长度时，只取能量最高的片段
        window_rms = np.zeros(windows)
        for i in range(windows):
            start = i * step_size
            end = min(start + window_size, len(x))
            segment = x[start:end]
            window_rms[i] = np.sqrt(np.mean(segment ** 2))
        max_idx = np.argmax(window_rms)
        sample_start = max_idx * step_size
        sample_end = min(len(x), sample_start + window_size)
        x = x[sample_start:sample_end]

    # 动态范围压缩
    if compressionStatus:
        x = dynamicRangeCompression(x, compressionThreshold, compressionRatio)
        print("压缩器已启用")
    else:
        print("压缩器未启用")

    # ========= SF 与 ZCR 起始点检测 =========
    onsetTime_SF, _, sf, t_SF, Threshold_SF = Onsetsdetect_SF(x, fs, threshold_SF, frequencyRange_SF)
    onsetTime_ZCR, _, zcr, t_ZCR, Threshold_ZCR = Onsetsdetect_ZCR(x, fs, threshold_ZCR, frequencyRange_ZCR)

    # ========= 计算 SF 的 BPM =========
    if len(onsetTime_SF) > 1:
        timeDiffs_SF = np.diff(onsetTime_SF)
        mean_diff_SF = np.mean(timeDiffs_SF)
        std_diff_SF = np.std(timeDiffs_SF)
        filtered_diffs_SF = timeDiffs_SF[np.abs(timeDiffs_SF - mean_diff_SF) <= Std_factor * std_diff_SF]

        if len(filtered_diffs_SF) > 0:
            avg_time_diff_SF = np.mean(filtered_diffs_SF)
            bpm_SF = 60 / avg_time_diff_SF
        else:
            bpm_SF = np.nan
    else:
        bpm_SF = np.nan
        filtered_diffs_SF = []

    # ========= 计算 ZCR 的 BPM =========
    if len(onsetTime_ZCR) > 1:
        timeDiffs_ZCR = np.diff(onsetTime_ZCR)
        mean_diff_ZCR = np.mean(timeDiffs_ZCR)
        std_diff_ZCR = np.std(timeDiffs_ZCR)
        filtered_diffs_ZCR = timeDiffs_ZCR[np.abs(timeDiffs_ZCR - mean_diff_ZCR) <= Std_factor * std_diff_ZCR]

        if len(filtered_diffs_ZCR) > 0:
            avg_time_diff_ZCR = np.mean(filtered_diffs_ZCR)
            bpm_ZCR = 60 / avg_time_diff_ZCR
        else:
            bpm_ZCR = np.nan
    else:
        bpm_ZCR = np.nan
        filtered_diffs_ZCR = []

    # ========= 计算均匀度 Uniformity =========
    if len(filtered_diffs_SF) > 0:
        std_diff_SF = np.std(filtered_diffs_SF)
        uniformity_SF = 1 / std_diff_SF if std_diff_SF > 0 else 0
    else:
        uniformity_SF = 0
        std_diff_SF = 0

    if len(filtered_diffs_ZCR) > 0:
        std_diff_ZCR = np.std(filtered_diffs_ZCR)
        uniformity_ZCR = 1 / std_diff_ZCR if std_diff_ZCR > 0 else 0
    else:
        uniformity_ZCR = 0
        std_diff_ZCR = 0

    # ========= 权重分配 =========
    if std_diff_SF == 0 and std_diff_ZCR == 0:
        w_SF, w_ZCR = 0.5, 0.5
    elif std_diff_SF == 0:
        w_SF, w_ZCR = 1, 0
    elif std_diff_ZCR == 0:
        w_SF, w_ZCR = 0, 1
    else:
        total_uniformity = uniformity_SF + uniformity_ZCR
        w_SF = uniformity_SF / total_uniformity
        w_ZCR = uniformity_ZCR / total_uniformity

    print(f"调整后 SF 权重: {w_SF:.3f}")
    print(f"调整后 ZCR 权重: {w_ZCR:.3f}")

    # ========= 修正极端值并计算加权 BPM =========
    def adjust_bpm(bpm):
        if np.isnan(bpm):
            return bpm
        if bpm < 20:
            return bpm * 8
        elif bpm < 40:
            return bpm * 4
        elif bpm < 80:
            return bpm * 2
        else:
            return bpm

    bpm_SF = adjust_bpm(bpm_SF)
    bpm_ZCR = adjust_bpm(bpm_ZCR)

    if np.isnan(bpm_SF):
        bpm = bpm_ZCR
    elif np.isnan(bpm_ZCR):
        bpm = bpm_SF
    else:
        bpm = w_SF * bpm_SF + w_ZCR * bpm_ZCR

    # ========= 绘图（可选） =========
    if ploteachsample:
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t_SF, sf, 'b')
        plt.scatter(onsetTime_SF, np.interp(onsetTime_SF, t_SF, sf), color='r', marker='*')
        plt.axhline(Threshold_SF, color='r', linestyle='--', linewidth=1.5)
        plt.title("SF方法 - 谱流检测")
        plt.xlabel("时间 (s)")
        plt.ylabel("Spectral Flux")

        plt.subplot(3, 1, 2)
        plt.plot(t_ZCR, zcr, 'b')
        plt.scatter(onsetTime_ZCR, np.interp(onsetTime_ZCR, t_ZCR, zcr), color='r', marker='*')
        plt.axhline(Threshold_ZCR, color='r', linestyle='--', linewidth=1.5)
        plt.title("ZCR方法 - 过零率检测")
        plt.xlabel("时间 (s)")
        plt.ylabel("Zero Crossing Rate")

        plt.subplot(3, 1, 3)
        plt.plot(x, 'k')
        plt.scatter(onsetTime_SF * fs, np.ones_like(onsetTime_SF), color='g', marker='*')
        plt.scatter(onsetTime_ZCR * fs, -np.ones_like(onsetTime_ZCR), color='m', marker='*')
        plt.title("SF 与 ZCR 起始点")
        plt.xlabel("采样点")
        plt.ylabel("幅度")
        plt.tight_layout()
        plt.show()

    return bpm
