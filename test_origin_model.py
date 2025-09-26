# cat pretrain_before.py 
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("./models/clouditera_SecGPT-1.5B", trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained("./models/clouditera_SecGPT-1.5B", trust_remote_code=True)

# 定制问答
questions = [
    {"question": "SecGPT-1.5B的作者是谁？", "label": "clouditera团队。"},
    {"question": "本地安全策略编号LS-2024-001是什么？", "label": "这是一个测试策略，用于演示LoRA微调效果。"},
    {"question": "CVE-2099-99999属于什么漏洞类型？", "label": "本地提权。"}
]

for item in questions:
    q = item["question"]
    label = item["label"]
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只保留模型生成的答案部分
    answer = answer.replace(q, "").strip()
    print(f"【问题】{q}")
    print(f"【标准答案】{label}")
    print(f"【模型输出】{answer}")
    print(f"【是否答对】{'✔️' if label in answer else '❌'}")
    print("="*50)

