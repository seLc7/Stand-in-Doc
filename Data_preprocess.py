import pandas as pd
import os
import random

# 读取上传的 Excel 文件
file_path = "SmoothNLP36kr新闻数据集10k.xlsx"
df = pd.read_excel(file_path)

# 确定包含文章内容的列（假设列名可能为 'content' 或类似名称）
content_column = None
for col in df.columns:
    if "content" in col.lower():
        content_column = col
        break

# 如果找到了内容列，则进行随机抽样和存储
if content_column:
    # 随机选择 10 篇文章
    sampled_articles = (
        df[content_column].dropna().sample(n=1000, random_state=7).tolist()
    )

    # 定义存储路径
    output_dir = os.path.join(os.getcwd(), "selected_articles")
    os.makedirs(output_dir, exist_ok=True)

    # 逐篇文章存储为 txt 文件
    file_paths = []
    for i, article in enumerate(sampled_articles):
        file_name = f"article_{i+1}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(article)
        file_paths.append(file_path)

    file_paths  # 返回生成的文件列表
else:
    content_column = df.columns  # 如果找不到内容列，返回所有列名以便检查
