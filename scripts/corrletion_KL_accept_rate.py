import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def compute_correlations(file_path1, file_path2):
    correlations_pearson = []
    correlations_spearman = []
    count_negative_pearson = 0
    count_positive_pearson = 0

    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
        
        for i, (line1, line2) in enumerate(zip(file1, file2)):
            data1 = np.fromstring(line1.strip(), sep=',')
            data2 = np.fromstring(line2.strip(), sep=',')

            # 处理数据长度不一致的情况
            if len(data1) != len(data2):
                print(f"Warning: 数据长度不一致 - 文件1长度为{len(data1)}, 文件2长度为{len(data2)}")
                min_length = min(len(data1), len(data2))
                data1 = data1[:min_length]
                data2 = data2[:min_length]

            # 替换无效值为10^10
            data1 = np.nan_to_num(data1, nan=1e10, posinf=1e10, neginf=-1e10)
            data2 = np.nan_to_num(data2, nan=1e10, posinf=1e10, neginf=-1e10)
            
            data1_shifted = data1[:-1]
            data2 = data2[1:]            

            correlation_pearson = pearsonr(data1_shifted, data2)[0]
            correlation_spearman = spearmanr(data1_shifted, data2)[0]

            correlations_pearson.append(correlation_pearson)
            correlations_spearman.append(correlation_spearman)
            
            if correlation_pearson < 0:
                count_negative_pearson += 1
            elif correlation_pearson > 0:
                count_positive_pearson += 1

    print("Number of Pearson correlation coefficients less than 0:", count_negative_pearson)
    print("Number of Pearson correlation coefficients greater than 0:", count_positive_pearson)   

    return correlations_pearson, correlations_spearman

file_path1 = '/work/valex1377/LLMSpeculativeSampling/kl_div_out.csv'
file_path2 = '/work/valex1377/LLMSpeculativeSampling/scripts/speculative_accepted_sequence_test_______.csv'

correlations_pearson, correlations_spearman = compute_correlations(file_path1, file_path2)

print("Pearson correlation coefficients:", correlations_pearson)
print("Spearman correlation coefficients:", correlations_spearman)



# 绘制相关系数图表
fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.35
index = np.arange(len(correlations_pearson))

bar1 = ax.bar(index, correlations_pearson, bar_width, label='Pearson Correlation')
bar2 = ax.bar(index + bar_width, correlations_spearman, bar_width, label='Spearman Correlation', color='orange')

ax.set_title('Correlation Analysis')
ax.set_xlabel('Row Index')
ax.set_ylabel('Correlation')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(index, fontsize=6)  # Adjust font size here
ax.legend()

plt.tight_layout()

# 保存图表为PNG文件
plt.savefig('correlation_bar_chart.png')
