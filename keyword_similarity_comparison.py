import os
import spacy
import re
import numpy as np
from urllib.parse import urlparse
from Levenshtein import ratio as levenshtein_ratio

nlp = spacy.load("zh_core_web_trf")  # 中文模型，或换英文模型 if EN


def extract_key_info(text):
    """
    从文本中提取关键信息，包括组织、地点、时间、网址、人名、金钱数额、百分比、数量、时间表达、序数和基数。

    参数:
    text (str): 输入的文本字符串。

    返回:
    dict: 包含提取信息的字典。
    """
    doc = nlp(text)
    info = {
        "组织": [],
        "地点": [],
        "时间": [],
        "网址": [],
        "人名": [],
        "金钱数额": [],
        "百分比": [],
        "数量": [],
        "时间表达": [],
        "序数": [],
        "基数": [],
    }

    for ent in doc.ents:
        if ent.label_ == "ORG":
            info["组织"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            info["地点"].append(ent.text)
        elif ent.label_ == "DATE":
            info["时间"].append(ent.text)
        elif ent.label_ == "PERSON":
            info["人名"].append(ent.text)
        elif ent.label_ == "MONEY":
            info["金钱数额"].append(ent.text)
        elif ent.label_ == "PERCENT":
            info["百分比"].append(ent.text)
        elif ent.label_ == "QUANTITY":
            info["数量"].append(ent.text)
        elif ent.label_ == "TIME":
            info["时间表达"].append(ent.text)
        elif ent.label_ == "ORDINAL":
            info["序数"].append(ent.text)
        elif ent.label_ == "CARDINAL":
            info["基数"].append(ent.text)

    # 自定义提取网址
    urls = re.findall(r"https?://[^\s)]+", text)
    info["网址"].extend(urls)

    # 去重
    for key in info:
        info[key] = list(set(info[key]))

    return info


# # 示例使用
# with open("selected_articles/used/article_1.txt", "r", encoding="utf-8") as f1:
#     text1 = f1.read()

# with open("tishen/m/article_1.txt", "r", encoding="utf-8") as f2:
#     text2 = f2.read()

# info1 = extract_key_info(text1)
# info2 = extract_key_info(text2)

# print("文档1实体：", info1)
# print("文档2实体：", info2)


def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = set(set1) & set(set2)
    union = set(set1) | set(set2)
    return len(intersection) / len(union)


def compute_entity_diff_ratio(entities1, entities2):
    all_keys = set(entities1.keys()) | set(entities2.keys())
    total_fields = len(all_keys)
    different_fields = 0

    for key in all_keys:
        set1 = set(entities1.get(key, []))
        set2 = set(entities2.get(key, []))
        if set1 != set2:
            different_fields += 1

    ratio = different_fields / total_fields if total_fields else 0
    return different_fields, total_fields, ratio


def compute_overall_jaccard_similarity(entities1, entities2):
    all_keys = set(entities1.keys()) | set(entities2.keys())
    total_sim = 0
    for key in all_keys:
        s1 = set(entities1.get(key, []))
        s2 = set(entities2.get(key, []))
        sim = jaccard_similarity(s1, s2)
        total_sim += sim
    return total_sim / len(all_keys) if all_keys else 1.0


def compute_average_levenshtein_similarity(entities1, entities2):
    all_keys = set(entities1.keys()) | set(entities2.keys())
    total_score = 0
    count = 0

    for key in all_keys:
        set1 = entities1.get(key, [])
        set2 = entities2.get(key, [])
        if not set1 and not set2:
            total_score += 1
            count += 1
        elif set1 and set2:
            for a in set1:
                max_sim = max(levenshtein_ratio(a, b) for b in set2)
                max_sim = max(0, min(1, max_sim))  # 确保相似度在 [0, 1] 范围内
                total_score += max_sim
                count += 1
        else:
            count += max(len(set1), len(set2))

    return total_score / count if count else 1.0


def dice_coefficient(set1, set2):
    """
    计算两个集合之间的 Dice 系数。

    参数:
    - set1: 第一个集合
    - set2: 第二个集合

    返回:
    - Dice 系数，范围在 [0,1] 之间
    """
    # 将输入转换为 NumPy 数组
    set1 = np.array(list(set1))
    set2 = np.array(list(set2))

    # 计算交集的大小
    intersection = len(np.intersect1d(set1, set2))
    total_elements = len(set1) + len(set2)

    if total_elements == 0:
        return 1.0  # 如果两个集合都为空，定义相似度为 1
    return 2 * intersection / total_elements


def compute_average_dice_similarity(entities1, entities2):
    """
    计算两个实体字典之间的平均 Dice 相似度。

    参数:
    - entities1: 第一个实体字典，键为实体类别，值为该类别下的实体列表。
    - entities2: 第二个实体字典，格式同上。

    返回:
    - 平均 Dice 相似度，值在 [0, 1] 之间。
    """
    all_keys = set(entities1.keys()) | set(entities2.keys())
    total_score = 0
    count = 0

    for key in all_keys:
        set1 = set(entities1.get(key, []))
        set2 = set(entities2.get(key, []))
        if not set1 and not set2:
            continue  # 如果两个集合都为空，跳过计算
        total_score += dice_coefficient(set1, set2)
        count += 1

    return total_score / count if count else 1.0


def compare_articles_in_dirs(dir1, dir2, output_report=True):
    results = []
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))

    common_files = set(files1) & set(files2)

    if not common_files:
        print("两个文件夹下没有同名文件")
        return

    for filename in common_files:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)

        if not os.path.isfile(path1) or not os.path.isfile(path2):
            # print(f"⏭️ 跳过非文件：{filename}")
            continue

        try:
            with open(path1, "r", encoding="utf-8") as f1, open(
                path2, "r", encoding="utf-8"
            ) as f2:
                text1 = f1.read()
                text2 = f2.read()

            info1 = extract_key_info(text1)
            info2 = extract_key_info(text2)

            diff_count, total_fields, diff_ratio = compute_entity_diff_ratio(
                info1, info2
            )
            jaccard_score = compute_overall_jaccard_similarity(info1, info2)
            levenshtein_score = compute_average_levenshtein_similarity(info1, info2)
            dice_score = compute_average_dice_similarity(info1, info2)

            result = {
                "文件名": filename,
                "字段差异率": f"{diff_ratio:.2%}",
                "Jaccard相似度": f"{jaccard_score:.2%}",
                "Levenshtein相似度": f"{levenshtein_score:.2%}",
                "Dice 相似度": f"{dice_score:.2%}",
            }
            results.append(result)

            print(f"\n📝 {filename}")
            for k, v in result.items():
                if k != "文件名":
                    print(f"  {k}: {v}")

        except Exception as e:
            print(f"❌ 处理文件 {filename} 出错: {e}")

    # 可选：写入CSV报告
    if output_report:
        import csv

        with open(
            "keyword_comparison_report.csv", "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "文件名",
                    "字段差异率",
                    "Jaccard相似度",
                    "Levenshtein相似度",
                    "Dice 相似度",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print("\n✅ 已生成差异对比报告：keyword_comparison_report.csv")


# 使用方式
compare_articles_in_dirs("selected_articles\gen_3", "tishen\gen_3")
