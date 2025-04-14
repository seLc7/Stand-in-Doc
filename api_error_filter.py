import os

# 指定要检查的文件夹路径
folder_path = "tishen"  # 请替换为你的实际路径

# 需要查找的错误信息
error_message = "API 调用失败！错误信息：Request timed out."

# 遍历文件夹中的所有 txt 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # 读取文件内容
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # 检查是否包含指定错误信息
            if error_message in content:
                os.remove(file_path)  # 删除文件
                print(f"已删除文件: {file_path}")

        except Exception as e:
            print(f"读取 {filename} 失败: {e}")

print("检查完成！")
