
import re

# 假设列表 S 中的元素是字符串形式
S = ["S['12,467']"]

# 正则表达式模式：匹配引号内的内容
pattern = r"'(.*?)'"

# 提取引号内的内容
result = [re.search(pattern, s).group(1) for s in S]

print(result)