import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_new = pd.read_csv("Updated_Composite_Similarity_gpt_gen2.1.csv")
# 设置绘图风格
sns.set(style="whitegrid")

# 1. 计算不同相似度指标的总体分布情况（箱线图）
# 计算箱线图关键数值
stats_summary = df_new[
    [
        "TF-IDF",
        "Jaccard",
        "LCS",
        "Levenshtein",
        "BERT",
        "Composite",
        "Composite_Zscore",
        "Composite_MinMax",
    ]
].describe()

# 打印箱线图的关键统计数据
print("Box Plot Statistics Summary:")
print(stats_summary)

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_new[
        [
            "TF-IDF",
            "Jaccard",
            "LCS",
            "Levenshtein",
            "BERT",
            # "Composite",
            # "Composite_Zscore",
            # "Composite_MinMax",
        ]
    ],
    width=0.5,
)
plt.title(
    "Distribution of Different Content Similarity Metrics",
    fontweight="bold",
    fontsize=16,
)
plt.ylabel("Content Similarity")
plt.ylim(0, None)  # <--- 强制y轴从0开始
new_xticklabels = [
    "TF-IDF",
    "Jaccard",
    "LCS",
    "Levenshtein",
    "BERT",
    # "Composite",
    # "Composite_Zscore",
    # "Composite_MinMax",
]
plt.xticks(
    # ticks=range(8),  # 对应数据框的6列位置（0到5）
    ticks=range(len(new_xticklabels)),  # 自动适应长度
    labels=new_xticklabels,  # 绑定自定义标签
    fontsize=14,
)
plt.yticks(
    fontsize=14,
)
plt.show()

# # 2. 绘制BERT相似度与其他指标的散点图，观察其相关性
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=df_new["BERT"], y=df_new["TF-IDF"], alpha=0.6)
# plt.title("BERT Similarity vs TF-IDF Similarity ")
# plt.xlabel("BERT Similarity ")
# plt.ylabel("TF-IDF Similarity ")
# plt.grid(True)
# plt.show()

# 3. 计算 Pearson / Spearman 相关性并绘制热力图
df_numeric = df_new.drop(columns=["File Names"])
correlation_pearson = df_numeric.corr(method="pearson")
correlation_spearman = df_numeric.corr(method="spearman")

# Pearson 相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_pearson, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(
    "Pearson Heatmap",
    fontweight="bold",
    fontsize=16,
)
plt.xticks(
    fontsize=14,
)
plt.yticks(
    fontsize=14,
)
plt.tight_layout()  # <--- 新增此行
plt.show()

# Spearman 相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_spearman, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
plt.title(
    "Spearman Heatmap",
    fontweight="bold",
    fontsize=16,
)
plt.xticks(
    fontsize=14,
)
plt.yticks(
    fontsize=14,
)
plt.tight_layout()  # <--- 新增此行
plt.show()
