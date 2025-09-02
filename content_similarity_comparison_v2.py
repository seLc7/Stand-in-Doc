import os
import re
import jieba
import pandas as pd
import multiprocessing
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio
from difflib import SequenceMatcher

# 设置文件夹路径
file_path_1 = "selected_articles"
file_path_2 = "tishen"

# 预加载 BERT 模型
bert_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def clean_text(text):
    """文本预处理：去除标点符号和特殊字符"""
    return re.sub(r"[^\w\s]", "", text).strip()


def tokenize(text):
    """中文分词"""
    return list(jieba.cut(clean_text(text)))


def calculate_tfidf_similarity(doc1, doc2, tfidf_vectorizer):
    """TF-IDF 余弦相似度（使用预训练 Vectorizer）"""
    tfidf_matrix = tfidf_vectorizer.transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


def calculate_jaccard_similarity(words1, words2):
    """计算 Jaccard 相似度（使用分词结果）"""
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union != 0 else 0


def calculate_lcs_similarity(doc1, doc2):
    """计算最长公共子序列（LCS）相似度"""
    seq_matcher = SequenceMatcher(None, doc1, doc2)
    return seq_matcher.ratio()


def calculate_bert_similarity(doc1, doc2):
    """BERT 语义相似度（使用预加载模型）"""
    embeddings = bert_model.encode([doc1, doc2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def calculate_levenshtein_similarity(doc1, doc2):
    """Levenshtein 结构相似度"""
    return ratio(doc1, doc2)


def calculate_combined_similarity(tfidf, jaccard, lcs, bert, levenshtein, weights=None):
    """综合相似度计算（加权平均）"""
    if weights is None:
        weights = [0.1, 0.2, 0.2, 0.3, 0.2]  # 默认权重
    return sum(w * s for w, s in zip(weights, [tfidf, jaccard, lcs, bert, levenshtein]))


def get_all_files(directory):
    """递归获取文件夹中的所有文件（包含子目录）"""
    all_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)  # 获取相对路径
            all_files[relative_path] = file_path
    return all_files


def process_file_pair(params):
    """多进程处理文件对的相似度计算"""
    filename, file1_path, file2_path, tfidf_vectorizer = params
    try:
        with open(file1_path, "r", encoding="utf-8") as f1, open(
            file2_path, "r", encoding="utf-8"
        ) as f2:
            doc1 = f1.read()
            doc2 = f2.read()

            # 预计算分词
            words1 = set(tokenize(doc1))
            words2 = set(tokenize(doc2))

            # 计算相似度
            tfidf_score = calculate_tfidf_similarity(doc1, doc2, tfidf_vectorizer)
            jaccard_score = calculate_jaccard_similarity(words1, words2)
            lcs_score = calculate_lcs_similarity(doc1, doc2)
            bert_score = calculate_bert_similarity(doc1, doc2)
            levenshtein_score = calculate_levenshtein_similarity(doc1, doc2)

            # 计算综合相似度
            combined_score = calculate_combined_similarity(
                tfidf_score, jaccard_score, lcs_score, bert_score, levenshtein_score
            )

            return (
                filename,
                tfidf_score,
                jaccard_score,
                lcs_score,
                bert_score,
                levenshtein_score,
                combined_score,
            )

    except Exception as e:
        print(f"⚠️ 处理文件 {filename} 时出错: {e}")
        return None


def main():
    # 获取两个子文件夹中的所有文件
    selected_files = get_all_files(file_path_1)
    tishen_files = get_all_files(file_path_2)

    # 找到同名文件（相对路径相同）
    common_files = set(selected_files.keys()).intersection(set(tishen_files.keys()))

    # 训练 TF-IDF 向量化器（仅一次）
    all_docs = [
        open(selected_files[f], "r", encoding="utf-8").read() for f in common_files
    ]
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    tfidf_vectorizer.fit(all_docs)

    # 计算相似度（使用多进程）
    print(f"\n🔍 发现 {len(common_files)} 个同名文件，开始处理...\n")

    params = [
        (f, selected_files[f], tishen_files[f], tfidf_vectorizer) for f in common_files
    ]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_file_pair, params),
                total=len(common_files),
                desc="计算相似度",
                unit="file",
            )
        )

    # 过滤 None 结果
    results = [r for r in results if r is not None]

    # 保存结果到 CSV
    df = pd.DataFrame(
        results,
        columns=[
            "文件名",
            "TF-IDF 相似度",
            "Jaccard 词相似度",
            "LCS 句子相似度",
            "BERT 语义相似度",
            "Levenshtein 结构相似度",
            "综合相似度",
        ],
    )
    df.to_csv("content_similarity.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 相似度分析完成，结果已保存\n")


if __name__ == "__main__":
    main()
