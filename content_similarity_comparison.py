import os
import re
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio
from difflib import SequenceMatcher
from tqdm import tqdm

# from gensim.models import KeyedVectors

# 文件夹路径
file_path_1 = "selected_articles/gen_1"
file_path_2 = "tishen/gen_1"


def clean_text(text):
    """文本预处理：去除标点符号和特殊字符"""
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def tokenize(text):
    """中文分词处理"""
    return list(jieba.cut(clean_text(text)))


# 1.TF-IDF余弦相似度
def calculate_tfidf_similarity(doc1, doc2):
    """TF-IDF余弦相似度计算"""
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


# 2.Jaccard相似度
def calculate_jaccard_similarity(doc1, doc2):
    """计算 Jaccard 相似度"""
    words1 = set(tokenize(doc1))
    words2 = set(tokenize(doc2))
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union != 0 else 0


# 3.LCS相似度
def calculate_lcs_similarity(doc1, doc2):
    """计算最长公共子序列（LCS）相似度"""
    seq_matcher = SequenceMatcher(None, doc1, doc2)
    return seq_matcher.ratio()


# 4.WMD相似度
def calculate_wmd_similarity(doc1, doc2, model_path="path/to/word2vec.model"):
    """计算 Word Mover’s Distance 相似度"""
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    words1 = tokenize(doc1)
    words2 = tokenize(doc2)
    return model.wmdistance(words1, words2)


# 5.依许句法相似度
def calculate_syntax_similarity(doc1, doc2):
    """计算基于依存句法的相似度"""
    parsed1 = nlp(doc1)
    parsed2 = nlp(doc2)
    return parsed1.similarity(parsed2)


# 6.BERT语义相似度
def calculate_bert_similarity(doc1, doc2):
    """BERT语义相似度计算"""
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embeddings = model.encode([doc1, doc2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


# 7.Levenshtein结构相似度
def calculate_levenshtein_similarity(doc1, doc2):
    """结构相似度计算"""
    return ratio(doc1, doc2)


# 计算综合相似度
def calculate_combined_similarity(tfidf, jaccard, lcs, bert, levenshtein, weights=None):
    """综合相似度计算（加权平均）"""
    if weights is None:
        weights = [0.1, 0.1, 0.2, 0.4, 0.2]  # 默认平均权重
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


def main():
    # 获取两个子文件夹中的所有文件
    selected_files = get_all_files(file_path_1)
    tishen_files = get_all_files(file_path_2)

    # 找到同名文件（相对路径相同）
    common_files = set(selected_files.keys()).intersection(set(tishen_files.keys()))

    # 计算相似度
    results = []
    print(f"\n🔍 发现 {len(common_files)} 个同名文件，开始处理...\n")

    for filename in tqdm(common_files, desc="正在计算相似度", unit="file"):
        file1 = selected_files[filename]
        file2 = tishen_files[filename]

        try:
            with open(file1, "r", encoding="utf-8") as f1, open(
                file2, "r", encoding="utf-8"
            ) as f2:
                text1 = f1.read()
                text2 = f2.read()

                # 计算相似度
                tfidf_score = calculate_tfidf_similarity(text1, text2)
                jaccard_score = calculate_jaccard_similarity(text1, text2)
                lcs_score = calculate_lcs_similarity(text1, text2)
                bert_score = calculate_bert_similarity(text1, text2)
                levenshtein_score = calculate_levenshtein_similarity(text1, text2)

                # 计算综合相似度
                combined_score = calculate_combined_similarity(
                    tfidf_score, jaccard_score, lcs_score, bert_score, levenshtein_score
                )

                # 记录结果
                results.append(
                    (
                        filename,
                        tfidf_score,
                        jaccard_score,
                        lcs_score,
                        bert_score,
                        levenshtein_score,
                        combined_score,
                    )
                )

        except Exception as e:
            print(f"⚠️ 处理文件 {filename} 时出错: {e}")

    # # 输出相似度结果
    # print("\n相似度分析报告（同名文件对比）:")
    # print(
    #     "文件名 | TF-IDF 余弦相似度 | Jaccard词相似度 | LCS相似度 | BERT语义相似度 | 结构相似度"
    # )
    # print("-" * 50)
    # for (
    #     filename,
    #     tfidf_score,
    #     jaccard_score,
    #     lcs_score,
    #     bert_score,
    #     levenshtein_score,
    # ) in results:
    #     print(
    #         f"{filename} | {tfidf_score:.3f} | {jaccard_score:.3f} | {lcs_score:.3f} | {bert_score:.3f} |{levenshtein_score:.3f}"
    #     )

    # 保存结果到 CSV 文件
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
    df.to_csv("content similarity comparison.csv", index=False, encoding="utf-8-sig")

    # 打印结果
    print("\n✅ 相似度分析完成，结果已保存\n")
    print(df)


if __name__ == "__main__":
    main()
