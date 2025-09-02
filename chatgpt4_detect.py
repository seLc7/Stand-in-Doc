import os
import json
import csv
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm


# ========================== 配置加载 ==========================
def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


config = load_config()

# 输入和输出配置
input_directory = os.path.join(os.getcwd(), "tishen")
output_csv = os.path.join(os.getcwd(), "chatgpt_analysis_results.csv")  # CSV输出路径


# ========================== OpenAI 客户端管理 ==========================
def create_client():
    """创建 OpenAI 客户端实例"""
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["api_base_url"].rstrip("/"),
        timeout=30,
    )


client = create_client()


# ========================== 核心功能 ==========================


def build_prompts(text, level="L2"):
    """构建 system prompt + user prompt"""

    if level == "L2":
        system_prompt = (
            "你是一个具备初步行业认知但不能访问任何外部事实信息的文本分析者，"
            "你不允许判断任何事实的真假，例如人物身份、公司历史、数据是否准确等。"
            "你只能判断文本是否在表达、术语、格式、结构上显得不自然或不专业。"
        )

        user_prompt = f"""
            请判断以下文本是否可能经过信息替换或非专业编辑：

            请特别注意：
            - 不要根据常识或公开知识判断文本中事实是否正确。
            - 只关注语句是否通顺、结构是否合理、术语是否自然。

            **文本内容：**
            {text}

            请分析：
            1. 文本表达是否显得突兀或混乱？
            2. 术语使用是否合理？排版是否像正式文档？
            3. 语气是否统一，是否像行业内的真实表达？

            请直接回答："文本自然" 或 "可能存在替换"，并简要说明理由。
            """

    elif level == "L3":
        system_prompt = (
            "你是一位文本分析专家，对行业背景、项目术语、人物信息具备较强识别与判断能力。"
            "请根据内容逻辑、语义合理性和专业背景，判断文本是否可能经过篡改。"
        )

        user_prompt = f"""
            请判断以下文本是否可能经过信息替换或伪造：

            **文本内容：**
            {text}

            请分析：
            1. 是否存在逻辑冲突、时间/人物不一致？
            2. 是否包含不合理表述或异常事件？
            3. 文本整体是否可信？

            请直接回答："文本自然" 或 "可能存在替换"，并简要说明理由。
            """
    else:
        raise ValueError("支持的级别为 L2 或 L3")

    return system_prompt, user_prompt


def analyze_text(text, level="L2"):
    """执行文本分析，支持 L2 / L3 模拟"""

    system_prompt, user_prompt = build_prompts(text, level)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content, "success"
    except Exception as e:
        return f"API 调用失败！错误信息：{e}", "error"


def process_file(input_file):
    """处理单个文件并返回结构化结果"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        analysis_result, status = analyze_text(text)

        return {
            "文件名": os.path.basename(input_file),
            "分析结果": analysis_result,
            "状态": status,
            "原始文本": text[:500] + "..." if len(text) > 500 else text,  # 保存部分文本
        }
    except Exception as e:
        return {
            "文件名": os.path.basename(input_file),
            "分析结果": f"文件处理失败：{str(e)}",
            "状态": "error",
            "原始文本": "",
        }


# ========================== 批处理与CSV输出 ==========================
def save_to_csv(results, csv_path):
    """将结果保存为CSV文件"""
    fieldnames = ["文件名", "状态", "分析结果", "原始文本"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def batch_process():
    """批量处理并保存CSV结果"""
    txt_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".txt")
    ]

    if not txt_files:
        print("⚠ 未找到任何 TXT 文件，请检查输入文件夹！")
        return

    print(f"🔄 发现 {len(txt_files)} 个文件，开始处理...")

    results = []
    for file in tqdm(txt_files, desc="Processing Files", unit="file"):
        results.append(process_file(file))

    save_to_csv(results, output_csv)
    print(f"✅ 处理完成！结果已保存至：{output_csv}")


if __name__ == "__main__":
    batch_process()
