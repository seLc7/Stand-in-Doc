import os
import json
from tqdm import tqdm
from openai import OpenAI

# from my_prompt import evaluation_prompt, title_extraction_prompt

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# ========================== 配置加载 ==========================


def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


config = load_config()

# 输入和输出文件夹
input_directory = os.path.join(os.getcwd(), "selected_articles")  # 读取的文件夹
output_directory = os.path.join(os.getcwd(), "tishen")  # 输出的文件夹

# 确保输出文件夹存在
os.makedirs(output_directory, exist_ok=True)


# ========================== OpenAI 客户端管理 ==========================


def create_client():
    """创建 OpenAI 客户端实例"""
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["api_base_url"].rstrip("/"),
        timeout=30,
    )


client = create_client()


# ========================== 程序主体 ================================
# 发送 Prompt 并获取 AI 回答
def get_chatgpt_response(prompt, model="gpt-4o"):

    SYSTEM_ROLE = """您是一个专业的结构化文本混淆工具，专门用于数据脱敏处理。
                    1. 严格遵循用户提供的混淆规则
                    2. 保持原文结构、逻辑和专业术语
                    3. 仅返回修改后的文本内容"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ROLE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 添加适度创造性
            top_p=0.9,
            # max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 调用失败！错误信息：{e}"


def structured_obfuscation(
    text: str, date_shift_range: int = 7, num_perturb_factor: float = 0.1
) -> str:
    """结构化文本混淆处理

    Args:
        text: 待处理文本
        date_shift_range: 日期偏移范围（天）
        num_perturb_factor: 数值扰动系数

    Returns:
        str: 替身文本
    """
    PROMPT_TEMPLATE = f"""
                    # 任务要求
                    任务是对文本进行关键信息，不修改任何其他内容，不增加摘要、不增加任何格式说明。
                    请严格按以下规则处理文档（仅替换指定信息，保持原文结构和内容）：

                    # 规则集：
                    ## 1. 实体替换
                    - 人名：同文化背景替换（知名人物替换为同领域人物）；知名人物（如政治人物、名人等）应替换为同一国籍或领域的人名；如果是特定角色（如职务），则可以保持职务不变，仅替换人名。
                    - 地名：层级替换（城市→城市，机构→机构）；如果是特定地点（如公司总部、学校名称等），替换为同类型地点，保持地理和文化的相似性；避免将敏感地名（如历史事件地名）替换为完全不同的地名，保持其在语境中的关联性；必须同步替换关联地标（如故宫→兵马俑时，太和殿→一号坑）。
                    
                    ## 2. 数值处理
                    - 普通数值：±{int(num_perturb_factor*100)}%随机扰动，并根据情况四舍五入至最接近的整数
                    - 货币金额：保持符号，±10%扰动，要确保货币符号和单位（如美元、人民币、欧元等）保持一致；确保货币符号、数量级（如千、万、百万等）与新的数值相匹配。
                    - 单位数值：保留单位，数值可在上下10%的范围内波动。
                    - 百分比数值：百分比应与其他数值一样，进行上下±10%的扰动，但保留百分号（%）和相应的数值格式。

                    ## 3. 时间混淆
                    - 日期：±{date_shift_range}天随机偏移，日期偏移应保持原有的格式
                    - 时间混淆应适应文档类型。例如，财务报告中的日期不应过于频繁地偏移，保持业务流程中的时间逻辑一致。
                    - 时间段：同步调整起止时间

                    ## 4. 敏感信息
                    - 身份证：替换为格式相同的随机身份证号，确保性别等信息一致，但数字完全替换。
                    - 银行卡/信用卡号：替换为有效格式的随机银行卡号，确保长度一致（如16位），并遵循校验规则（Luhn算法）。
                    - 电话/手机号码：电话号码替换为相同格式的随机号码，确保国际区号和本地号码段合理。
                    - 邮箱：替换为合法格式的随机邮箱，域名和用户名部分应被替换。可将用户名替换为类似结构的随机字符，确保仍然看起来像一个合法的邮箱地址。域名替换可以为某个常见的域名。

                    # 保留要求
                    ✅ 必须保留：
                    - 原文结构、缩进、换行
                    - 专业术语和逻辑推理

                    ❌ 禁止：
                    - 添加任何说明性文字
                    - 修改原文观点或段落顺序
                    - 引入新内容或个人见解

                    # 输出格式
                    直接返回修改后的完整文本，不要包含任何额外内容！

                    # 待处理文本
                    {text}
                    
                    """
    return get_chatgpt_response(PROMPT_TEMPLATE)


def process_file(input_file):
    """读取文件，处理内容，并保存"""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    modified_text = structured_obfuscation(text)

    if modified_text:
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(modified_text)


def batch_process():
    """批量处理 selected_articles 目录下的所有 .txt 文件，并添加进度条"""
    txt_files = [f for f in os.listdir(input_directory) if f.endswith(".txt")]

    if not txt_files:
        print("⚠ 未找到任何 TXT 文件，请检查 selected_articles 文件夹！")
        return

    print(f"🔄 发现 {len(txt_files)} 个文件，开始处理...")

    # 使用 tqdm 进度条
    for file in tqdm(txt_files, desc="Processing Files", unit="file"):
        process_file(os.path.join(input_directory, file))

    print("✅ 批量处理完成，所有文件已存入 'tishen' 目录！")


if __name__ == "__main__":
    batch_process()
    # try:
    #     models = client.models.list()  # 获取可用模型列表
    #     print("API 连接成功！可用模型列表：")
    #     for model in models.data:
    #         print(model.id)
    # except Exception as e:
    #     print("API 连接失败！错误信息：", e)
