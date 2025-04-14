# plot_keyword_similarity_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# 📂 数据读取与预处理
# ---------------------------------------
df = pd.read_csv("keyword_comparison_report_gen3(chatgpt).csv")


def percent_to_float(s):
    return float(s.strip("%"))


df["字段差异率_float"] = df["字段差异率"].apply(percent_to_float)
df["Jaccard_float"] = df["Jaccard相似度"].apply(percent_to_float)
df["Levenshtein_float"] = df["Levenshtein相似度"].apply(percent_to_float)
df["Dice_float"] = df["Dice 相似度"].apply(percent_to_float)

# 差异度评分（新版）
df["差异度评分"] = (
    0.35 * (df["字段差异率_float"] / 100)
    + 0.25 * (1 - df["Jaccard_float"] / 100)
    + 0.15 * (1 - df["Levenshtein_float"] / 100)
    + 0.25 * (1 - df["Dice_float"] / 100)
)

# ---------------------------------------
# 🎨 样式设定（ggplot风格 + seaborn 白底）
# ---------------------------------------
plt.style.use("ggplot")
sns.set_style(
    "whitegrid",
    {"grid.color": ".85", "axes.facecolor": "white", "axes.edgecolor": ".6"},
)
colors = ["#E64B35", "#4DBBD5", "#00A087"]
# ---------------------------------------
# 📈 差异度 vs 各指标散点图
# ---------------------------------------
plt.figure(figsize=(12, 6))
# plt.scatter(
#     df["字段差异率_float"], df["差异度评分"], label="Field Difference Rate", alpha=0.7
# )
plt.scatter(
    100 - df["Jaccard_float"],
    df["差异度评分"],
    label="Jaccard Similarity",
    alpha=0.7,
)
plt.scatter(
    100 - df["Levenshtein_float"],
    df["差异度评分"],
    label="Levenshtein Similarity",
    alpha=0.7,
)
plt.scatter(
    100 - df["Dice_float"], df["差异度评分"], label="Dice Similarity", alpha=0.7
)

# plt.title(
#     "Key-info Similarity Indicators vs. Adjusted Similarity Score",
#     fontweight="bold",
#     fontsize=16,
# )
plt.xlabel("Similarity Indicators (%)", fontsize=20, fontweight="bold")
plt.ylabel("Adjusted Similarity Score", fontsize=20, fontweight="bold")
plt.xticks(
    fontsize=16,
)
plt.yticks(
    fontsize=16,
)
plt.ylim(0, 1)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# # ---------------------------------------
# # 📊 KMeans 聚类分析（基于相似度指标）
# # ---------------------------------------
# features = df[["字段差异率_float", "Jaccard_float", "Levenshtein_float", "Dice_float"]]
# X_scaled = StandardScaler().fit_transform(features)

# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# df["cluster"] = kmeans.fit_predict(X_scaled)

# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     x=100 - df["Jaccard_float"],
#     y=df["字段差异率_float"],
#     hue=df["cluster"],
#     palette="Set2",
# )
# plt.title("KMeans Clustering of Document Difference Patterns")
# plt.xlabel("1 - Jaccard Similarity (%)")
# plt.ylabel("Field Difference Rate (%)")
# plt.legend(title="Cluster")
# plt.tight_layout()
# plt.show()

# # ---------------------------------------
# # 📉 差异评分分布：直方图 + 箱型图
# # ---------------------------------------
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# sns.histplot(df["差异度评分"], bins=30, kde=True, color="steelblue")
# plt.title("Distribution of Adjusted Difference Score")
# plt.xlabel("Score")
# plt.ylabel("Document Count")

# plt.subplot(1, 2, 2)
# sns.boxplot(y=df["差异度评分"], color="lightcoral")
# plt.title("Boxplot of Adjusted Difference Score")
# plt.ylabel("Score")

# plt.tight_layout()
# plt.show()

# ---------------------------------------
# 📉 PCA + t-SNE 降维可视化
# ---------------------------------------
df["结构分布差"] = df["字段差异率_float"] / 100  # 简化

X_all = df[
    [
        "字段差异率_float",
        "Jaccard_float",
        "Levenshtein_float",
        "Dice_float",
        "差异度评分",
    ]
]
X_scaled = StandardScaler().fit_transform(X_all)

# PCA
pca = PCA(n_components=2)
df[["PCA1", "PCA2"]] = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
df[["TSNE1", "TSNE2"]] = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))

# plt.subplot(1, 2, 1)
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue=df["差异度评分"],
    palette="coolwarm",
    data=df,
    marker="X",
    s=50,
    legend=False,
)
plt.title("PCA Projection Colored by Adjusted Similarity Score", fontweight="bold")
plt.xlabel("PCA1", fontsize=14, fontweight="bold")
plt.ylabel("PCA2", fontsize=14, fontweight="bold")
plt.xticks(
    fontsize=14,
)
plt.yticks(
    fontsize=14,
)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
# plt.subplot(1, 2, 2)
sns.scatterplot(
    x="TSNE1",
    y="TSNE2",
    hue=df["差异度评分"],
    palette="coolwarm",
    data=df,
    marker="X",
    s=50,
    legend=False,
)
plt.title("t-SNE Projection Colored by Adjusted Similarity Score", fontweight="bold")
plt.xlabel("t-SNE1", fontsize=14, fontweight="bold")
plt.ylabel("t-SNE2", fontsize=14, fontweight="bold")
plt.xticks(
    fontsize=14,
)
plt.yticks(
    fontsize=14,
)
plt.tight_layout()
plt.show()
