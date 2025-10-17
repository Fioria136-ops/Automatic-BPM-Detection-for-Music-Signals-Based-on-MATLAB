import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import traceback

# 将 Functions 加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

# 导入自定义函数
from analyzeBPM_v12 import analyzeBPM_v12
from extractRealBPM import extractRealBPM

try:
    # ========== 参数设置 ==========
    sampleDuration = 8
    frequencyRange_SF = [500, 7000]
    frequencyRange_ZCR = [2500, 18000]
    threshold_SF = 1.0
    threshold_ZCR = 1.0
    std_factor = 1.5
    compressionStatus = True
    plot_each_sample = False
    plot_result = True
    compressionThreshold = 0.1
    compressionRatio = 2.1
    tolerance = 15
    accuracy = 3  # 1/2/3 验证方式

    # ========== 路径设置 ==========
    script_path = os.path.dirname(os.path.abspath(__file__))
    audio_folder_path = os.path.join(script_path, 'Dataset')
    result_file_path = os.path.join(script_path, 'results', 'Sound_Analysis_Results.txt')
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    audio_files = glob.glob(os.path.join(audio_folder_path, '*.wav'))
    if not audio_files:
        input("未找到音频文件。请将 .wav 文件放入 Dataset 文件夹后按回车退出...")
        sys.exit()

    result_file = open(result_file_path, 'w', encoding='utf-8')
    result_file.write("音频分析结果\n=============================\n")

    y_true_all, y_pred_all = [], []
    total_deviation_rate = 0
    valid_file_count = 0

    # ========== 初始化绘图 ==========
    if plot_result:
        plt.figure()
        x = np.arange(115, 146)
        plt.plot(x, x/2, '--')
        plt.plot(x, x/1.5, '--')
        plt.plot(x, x, '-', linewidth=1.5)
        plt.plot(x, 1.5*x, '--')
        plt.plot(x, 2*x, '--')
        plt.title("BPM预测 vs 真实BPM")
        plt.xlabel("真实 BPM")
        plt.ylabel("预测 BPM")

    # ========== 批量处理音频 ==========
    for audio_file_path in audio_files:
        audio_file_name = os.path.basename(audio_file_path)
        print(f"正在分析 {audio_file_name} ...")

        try:
            bpm = analyzeBPM_v12(
                audio_file_path, sampleDuration, threshold_SF,
                compressionThreshold, compressionRatio, compressionStatus,
                plot_each_sample, std_factor, threshold_ZCR,
                frequencyRange_SF, frequencyRange_ZCR
            )

            realBPM1, realBPM2, realBPM3, realBPM4, realBPM5 = extractRealBPM(audio_file_name)
            if np.isnan(realBPM1):
                print(f"警告：无法提取真实 BPM —— {audio_file_name}")
                result_file.write(f"警告：无法提取真实 BPM —— {audio_file_name}\n\n")
                continue

            # 选择最接近的真实 BPM
            if accuracy == 3:
                bpm_candidates = [realBPM1, realBPM2, realBPM3, realBPM4, realBPM5]
            elif accuracy == 2:
                bpm_candidates = [realBPM1, realBPM3, realBPM5]
            else:
                bpm_candidates = [realBPM3]

            closest_real_bpm = bpm_candidates[np.argmin(np.abs(np.array(bpm_candidates) - bpm))]

            # 计算误差率
            deviation_rate = abs(bpm - closest_real_bpm) / closest_real_bpm * 100
            total_deviation_rate += deviation_rate
            valid_file_count += 1

            y_pred = 1 if abs(bpm - closest_real_bpm) <= tolerance else 0
            y_true = 1

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

            # 绘图
            if plot_result:
                color = 'r' if abs(bpm - closest_real_bpm) > tolerance else 'k'
                plt.plot(realBPM3, bpm, 'o', markeredgecolor=color, markersize=5, linewidth=1.5)

            # 输出结果
            print(f"文件: {audio_file_name}")
            print(f"预测 BPM: {bpm:.2f}, 真实 BPM: {realBPM3:.2f}, 最接近真实 BPM: {closest_real_bpm:.2f} (偏差: {deviation_rate:.2f}%)")
            result_file.write(f"文件: {audio_file_name}\n预测 BPM: {bpm:.2f}\n真实 BPM: {realBPM3:.2f}\n最接近真实 BPM: {closest_real_bpm:.2f} (偏差: {deviation_rate:.2f}%)\n\n")

        except Exception as e:
            print(f"分析 {audio_file_name} 时出错: {str(e)}")
            traceback.print_exc()
            result_file.write(f"分析 {audio_file_name} 时出错: {str(e)}\n\n")

    # ========== 平均偏差率 ==========
    average_deviation_rate = total_deviation_rate / valid_file_count if valid_file_count > 0 else np.nan
    print(f"平均偏差率: {average_deviation_rate:.2f}%")
    result_file.write(f"平均偏差率: {average_deviation_rate:.2f}%\n")
    result_file.close()

    # ========== 绘制最终图表 ==========
    if plot_result:
        plt.grid(True)
        plt.show()

    # ========== 混淆矩阵 ==========
    if y_true_all:
        conf_mat = confusion_matrix(y_true_all, y_pred_all)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["False(0)","True(1)"])
        disp.plot(cmap="Blues")
        plt.title("二分类混淆矩阵")
        plt.show()
        print("验证结果(FN,TP) =")
        print(conf_mat)

    input("分析完成，按回车退出...")

except Exception as e:
    print("脚本执行出现错误:", str(e))
    traceback.print_exc()
    input("按回车退出...")
