import requests
import json
from openai import OpenAI

# 使用requests库进行请求

# url = "http://localhost:8000/v1/completions"
# headers = {"Content-Type": "application/json"}
# data = {
#     "model": "./models/clouditera_SecGPT-1.5B",
#     "prompt": "什么是 XSS 攻击？",
#     "max_tokens": 100,
#     "temperature": 0.7
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(json.dumps(response.json(), indent=2, ensure_ascii=False))




# 使用OpenAI SDK进行请求

# client = OpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="token-abc123"  # 这里的API密钥可以是任意非空值
# )

# response = client.completions.create(
#     model="./models/clouditera_SecGPT-1.5B",
#     prompt="什么是 XSS 攻击？",
#     max_tokens=100
# )

# print(response.choices[0].text)





# 使用curl进行请求

# curl http://localhost:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "./models/clouditera_SecGPT-1.5B",
#     "prompt": "什么是 XSS 攻击？",
#     "max_tokens": 100
#   }'





