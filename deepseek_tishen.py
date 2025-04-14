import os
import requests
import time
import json
from base64 import b64encode
from tqdm import tqdm 

# 服务配置
API_URL = "http://58.206.234.157:11434/api/generate"
EMAIL = "testblockc@163.com"
PASSWORD = "llm0731zgc"



# 生成Basic Auth凭证
credentials = b64encode(f"{EMAIL}:{PASSWORD}".encode()).decode()
HEADERS = {"Authorization": f"Basic {credentials}", "Content-Type": "application/json"}

# 输入和输出文件夹
input_directory = os.path.join(os.getcwd(), "selected_articles")  # 读取的文件夹
output_directory = os.path.join(os.getcwd(), "tishen")  # 输出的文件夹

# 确保输出文件夹存在
os.makedirs(output_directory, exist_ok=True)


def call_deepseek_stream(prompt):
    data = {"model": "deepseek-r1:70b", "prompt": prompt}

    # 发送请求，并启用流式响应
    response = requests.post(API_URL, json=data, headers=HEADERS, stream=True)

    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
        return None

    full_response = ""  # 存储完整回答

    # 逐行读取流式返回的数据
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))  # 解析 JSON，修正错误
                chunk = json_data.get("response", "")  # 获取 response 字段
                print(chunk, end="", flush=True)  # 实时打印（可选）
                full_response += chunk  # 拼接完整响应
            except json.JSONDecodeError as e:
                print(f"\nJSON 解析错误: {e}, 原始数据: {line.decode('utf-8')}")

    print("\n")  # 换行
    return full_response


def get_deepseek_response(prompt):
    payload = {
        "model": "deepseek-r1:70b",
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
        response.raise_for_status()

        # 获取原始响应内容
        result = response.json().get("response")

        # 新增处理：清除<think>部分
        if result and "<think>" in result:
            # 分割并取最后一个元素，然后去除空白
            result = result.split("</think>")[-1].strip()
            # 可选：清除可能残留的<think>标签
            result = result.replace("<think>", "")

        return result

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print("原始响应内容:", response.text)
        return None
    except Exception as e:
        print(f"未知错误: {type(e).__name__}: {e}")
        return None


def structured_obfuscation(text, date_shift_range=7, num_perturb_factor=0.1):
    prompt = f"""
    执行文档关键信息替换，按照下面的规则来执行：
    
    <<规则集>>
    1. 实体替换：人名→同语系人名，地名→同国家地名，避免造成文化背景的不一致。
       人名：
       - 替换时，确保替换后的名字来自相同的文化背景或语言系，避免出现文化不符的情况。
       - 知名人物（如政治人物、名人等）应替换为同一国籍或领域的人名。
       - 如果是特定角色（如职务），则可以保持职务不变，仅替换人名。
       地点：
       - 如果是特定地点（如公司总部、学校名称等），替换为同类型地点，保持地理和文化的相似性。
       - 避免将敏感地名（如历史事件地名）替换为完全不同的地名，保持其在语境中的关联性。
       - 必须同步替换关联地标（如故宫→兵马俑时，太和殿→一号坑）
       - [强制] 保持地理逻辑链完整：
           ▪ 城市级替换
           ▪ 机构级替换
           ▪ 建筑级替换
           例如，城市北京被替换为西安后，其地标机构故宫博物馆替换为兵马俑博物馆，建筑太和殿替换为一号坑，只是举例，不一定非要替换成西安、兵马俑，替换成其他类似的也可以。
       
    
    2. 数值变换：所有数字上下{int(num_perturb_factor*100)}%扰动后四舍五入，忽略对日期或时间的数字修改。
       一般数值：
       - 对于一般数值，将上下浮动{int(num_perturb_factor*100)}%，并四舍五入至最接近的整数。
       - 对于数值较小（例如1-100）的扰动，可能需要更精细的比例，防止变动幅度过大。
       货币数值：
       - 对于货币金额，允许上下±10%的扰动，但要确保货币符号和单位（如美元、人民币、欧元等）保持一致。
       - 替换时，确保货币符号、数量级（如千、万、百万等）与新的数值相匹配。
       数量单位：
       - 对于带有单位（如千克、米、毫升等）的数值，单位不变，数值可在上下10%的范围内波动。
       - 对于特定单位（例如“厘米”和“米”），应根据转换规则进行必要的转换，保证数值的合理性。例如：“10厘米”可能会变为“11厘米”或“9厘米”，而不是将其替换为“0.1米”。
       百分比数值：
       - 百分比应与其他数值一样，进行上下±10%的扰动，但保留百分号（%）和相应的数值格式。

    3. 时间混淆：日期±{date_shift_range}天随机偏移，确保日期格式不被破坏。
       - 日期偏移应保持原有的格式，±{date_shift_range}天的随机变化。
       - 日期格式保持一致，如ISO 8601格式（YYYY-MM-DD），或其他常见格式。
       - 时间（如时分秒）偏移范围可调整，确保时效性不丢失。
       - 时间混淆应适应文档类型。例如，财务报告中的日期不应过于频繁地偏移，保持业务流程中的时间逻辑一致。
       - 在处理文档中的日期和时间时，确保理解上下文。对于带有具体时间（如“2025年3月1日 14:30”）的日期，可以通过±1小时或±15分钟的随机扰动保持时效性。
       - 对于跨越时间区间（例如“2025年1月1日到2025年1月31日”），应同时调整开始和结束日期，以避免产生不合理的日期差异。

    4. 敏感数据处理：
       - 身份证号：替换为格式相同的随机身份证号，确保出生日期、性别等信息一致，但数字完全替换。
       - 银行卡号/信用卡号**：替换为有效格式的随机银行卡号，确保长度一致（如16位），并遵循校验规则（Luhn算法）。
       - 电话号码：替换为相同格式的随机号码，确保国际区号和本地号码段合理。
       - 邮箱地址：替换为合法格式的随机邮箱，域名和用户名部分应被替换。可将用户名替换为类似结构的随机字符，确保仍然看起来像一个合法的邮箱地址。域名替换可以为某个常见的域名。
       - 地址信息：替换地址时，保持相同格式和类型（如街道名、城市、州、邮政编码），确保地址结构的合理性、逻辑性。

    5. 保留格式：维持原始文本中的JSON/XML/YAML结构，确保没有语法错误。
       - 替换时确保结构不被破坏，保留JSON的键值对顺序、XML的标签、YAML的缩进等。
       - 保证文档中的计算公式、逻辑关系和语法结构不受影响。

    <<输入文档>>
    {text}

    <<输出要求>>
    仅返回修改后的内容，保留原始缩进、换行等格式。若遇到无替代项时，保留原文。
    要求输出原文，即使是长文也要输出替换后的原文。
    """

    # return call_deepseek_stream(prompt)
    return get_deepseek_response(prompt)


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
