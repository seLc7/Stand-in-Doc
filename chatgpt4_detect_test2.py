import os
import openai
import json
import pandas as pd

# 设置 OpenAI API Key
openai.api_key = "your_openai_api_key"

# 指定存储文档的文件夹
articles_folder = "articles"


# 读取所有 txt 文件
def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):  # 只处理 .txt 文件
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append({"id": filename, "text": content})
    return documents


# 加载文件
documents = load_documents(articles_folder)


# 定义 API 请求函数
def analyze_document(document_text):
    """使用 ChatGPT API 进行文档语义和逻辑分析"""
    prompt = f"""
    你是一个专业的文本审核助手。请对以下文档进行分析，并回答以下问题：
    
    **文档内容：**
    {document_text}
    
    ### 任务：
    1. **语义检查**：这篇文档是否通顺、是否有明显逻辑错误或不连贯的地方？
    2. **一致性检查**：文档中的数据、时间、人物等信息是否一致？是否存在前后矛盾？
    3. **异常检测**：文档中是否存在突兀的、看似被替换或修改的内容？例如某个句子语义不合、上下文不匹配、风格不一致等？
    4. **潜在改动区域**（如有）：如果文档可能被修改，请指出可能的改动部分，并解释为何可能是改动内容。
    
    请用简明的格式返回你的分析结果。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "你是一个专业的文本审核助手"},
            {"role": "user", "content": prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]


# 遍历所有文档并处理
results = []
for doc in documents:
    result = analyze_document(doc["text"])
    results.append({"id": doc["id"], "analysis": result})

# # 保存结果到 JSON 文件
# with open("analysis_results.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# print("所有文档分析完成，结果已保存至 analysis_results.json")
df = pd.DataFrame(results)
df.to_csv("analysis_results.csv", index=False)
