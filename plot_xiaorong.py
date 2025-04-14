import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter

# 自动加载常见中文字体（兼容不同系统）
font_list = fm.findSystemFonts(fontpaths=None, fontext="ttf")
for font in font_list:
    if "SimHei" in font or "Microsoft YaHei" in font or "SimHei" in font:
        rcParams["font.sans-serif"] = [fm.FontProperties(fname=font).get_name()]
        break
rcParams["axes.unicode_minus"] = False

# 读取csv
df = pd.read_csv("消融.csv")
df["测试集名称"] = df["测试集名称"].fillna(method="ffill")


def percentage_to_float(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%", "")) / 100
    return x


cols_to_convert = [
    "RPR@Low-KA",
    "RPR@High-KA",
    "低水平通过率下降率",
    "高水平通过率下降率",
]
for col in cols_to_convert:
    df[col] = df[col].apply(percentage_to_float)

# 设置排序
custom_order = [
    "新闻类1 82 chatgpt",
    "新闻类2 82 deepseek",
    "新闻类1 64 chatgpt",
    "新闻类2 64 deepseek",
    "报告类1 82 chatgpt",
    "报告类2 82 deepseek",
    "报告类3 64 chatgpt",
    "报告类4 64 deepseek",
    "论文摘要类1 82 chatgpt",
    "论文摘要类2 82 deepseek",
    "论文摘要类3 64 chatgpt",
    "论文摘要类4 64 deepseek",
]

# 自定义横坐标顺序（实验条件）
column_order = [
    "Full Model",
    "NoDynNoise",
    "NoSurface",
    "Full Ablation",
]


df["测试集名称"] = pd.Categorical(
    df["测试集名称"], categories=custom_order, ordered=True
)

name_map = {
    "新闻类1 82 chatgpt": "[EN]News_8:2@ChatGPT",
    "新闻类2 82 deepseek": "[EN]News_8:2@DeepSeek",
    "新闻类1 64 chatgpt": "[CN]News_6:4@ChatGPT",
    "新闻类2 64 deepseek": "[CN]News_6:4@DeepSeek",
    "报告类1 82 chatgpt": "[EN]GovReport_8:2@ChatGPT",
    "报告类2 82 deepseek": "[EN]GovReport_8:2@DeepSeek",
    "报告类3 64 chatgpt": "[CN]GovReport_6:4@ChatGPT",
    "报告类4 64 deepseek": "[CN]GovReport_6:4@DeepSeek",
    "论文摘要类1 82 chatgpt": "[EN]PaperAbstract_8:2@ChatGPT",
    "论文摘要类2 82 deepseek": "[EN]PaperAbstract_8:2@DeepSeek",
    "论文摘要类3 64 chatgpt": "[CN]PaperAbstract_6:4@ChatGPT",
    "论文摘要类4 64 deepseek": "[CN]PaperAbstract_6:4@DeepSeek",
}


# 低水平通过率热力图数据
heatmap_data_low = df.pivot_table(
    index="测试集名称", columns="实验条件", values="RPR@Low-KA", aggfunc="mean"
).loc[custom_order, column_order]


heatmap_data_low.index = heatmap_data_low.index.to_series().map(name_map)
# 绘制美化版低水平热力图
plt.figure(figsize=(14, 8))
sns.set(font_scale=1.1)
ax = sns.heatmap(
    heatmap_data_low,
    annot=True,
    fmt=".4f",
    cmap="BuGn",  # 低水平推荐绿色系
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 16},
)
ax.set_xlabel("")
ax.set_ylabel("")
# plt.title(
#     "Realism Pass Rate at L1 Attackers",
#     fontsize=18,
#     fontweight="bold",
#     pad=20,
# )
# plt.ylabel("Test Set Name", fontsize=16, fontweight="bold", color="#555555")
# plt.xlabel("Experimental Scenario", fontsize=16, fontweight="bold", color="#555555")

plt.xticks(rotation=0, fontsize=20, style="italic", color="#555555")
plt.yticks(rotation=0, fontsize=20, color="#555555")
plt.tight_layout()
plt.show()

# 聚合数据
heatmap_data = df.pivot_table(
    index="测试集名称", columns="实验条件", values="RPR@High-KA", aggfunc="mean"
).loc[custom_order, column_order]
heatmap_data.index = heatmap_data.index.to_series().map(name_map)
# 绘制
plt.figure(figsize=(14, 8))
sns.set(font_scale=1.1)
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".4f",
    cmap="Blues",
    linewidths=0.5,  # 边框
    linecolor="white",  # 边框颜色
    cbar_kws={"shrink": 0.8},  # 调整色条大小
    annot_kws={"size": 16},  # 数字大小
)
ax.set_xlabel("")
ax.set_ylabel("")

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
# plt.title(
#     "Realism Pass Rate at L2 Attackers",
#     fontsize=18,
#     fontweight="bold",
#     pad=20,
# )
# plt.ylabel("Test Set Name", fontsize=16, fontweight="bold", color="#555555")
# plt.xlabel("Experimental Scenario", fontsize=16, fontweight="bold", color="#555555")
plt.xticks(rotation=0, fontsize=20, style="italic", color="#555555")
plt.yticks(rotation=0, fontsize=20, color="#555555")
plt.tight_layout()
plt.show()
