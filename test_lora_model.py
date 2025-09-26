from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 用于加载LoRA权重
import torch

# --------------------------
# 1. 配置路径（关键！需与微调时一致）
# --------------------------
base_model_path = "models/clouditera_SecGPT-1.5B"  # 原始基础模型路径（微调时用的模型）
lora_weights_path = "lora_secgpt_weights"          # 微调后保存的LoRA权重路径（微调代码第6步的输出）

# --------------------------
# 2. 加载原始基础模型和分词器（与微调时参数一致）
# --------------------------
# 加载基础模型（需保持与微调时相同的量化配置）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_4bit=True,          # 与微调时一致：4bit量化
    device_map="auto",          # 自动分配GPU/CPU（若需强制用0号GPU，设为device_map={"": 0}）
    torch_dtype=torch.bfloat16, # 与微调时一致：BF16精度
    trust_remote_code=True      # 信任远程代码（SecGPT-1.5B可能需要）
)

# 加载配套分词器（与微调时一致）
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
# 关键：保持与微调时相同的pad_token配置
tokenizer.pad_token = tokenizer.eos_token

# --------------------------
# 3. 加载LoRA权重，与基础模型合并（核心步骤）
# --------------------------
# PeftModel.from_pretrained：将LoRA权重“附着”到基础模型上
fine_tuned_model = PeftModel.from_pretrained(
    model=base_model,           # 原始基础模型
    model_id=lora_weights_path  # LoRA权重路径
)

# （可选）设置模型为评估模式（关闭dropout等训练特有的层，推理更稳定）
fine_tuned_model.eval()

# --------------------------
# 4. 测试生成（与微调代码的测试逻辑一致，可按需修改）
# --------------------------
def generate_answer(question, max_new_tokens=50):
    # 1. 构造与微调时相同格式的prompt（必须一致！否则模型无法理解任务）
    prompt = f"Instruction: {question}\nInput: \nOutput:"
    
    # 2. 分词：将prompt转换为模型可识别的input_ids（无需padding，推理时单条处理更高效）
    inputs = tokenizer(
        prompt,
        return_tensors="pt",  # 返回PyTorch张量
        truncation=True,      # 若prompt过长，截断到模型最大长度
        max_length=512        # 与微调时一致的最大长度
    ).to(fine_tuned_model.device)  # 将张量移到模型所在设备（GPU/CPU）
    
    # 3. 推理生成（关闭梯度计算，节省显存+加速）
    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # 最大生成token数（按需调整，如100）
            do_sample=False,                # 关闭随机采样：用贪心搜索（结果可复现）
            num_beams=1,                    #  beam search数量（=1时等价于贪心）
            pad_token_id=tokenizer.pad_token_id,  # 避免生成时出现警告
            eos_token_id=tokenizer.eos_token_id   # 生成到eos_token时停止
        )
    
    # 4. 解码：将模型输出的token_id转换为文本
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 5. 提取“Output:”后的答案（与微调时的prompt格式对应）
    if "Output:" in full_response:
        answer = full_response.split("Output:")[-1].strip()
    else:
        answer = full_response  # 异常情况：返回完整响应
    
    return answer

# --------------------------
# 5. 测试示例
# --------------------------
test_questions = [
    "SecGPT-1.5B的作者是谁？",
    "本地安全策略编号LS-2024-001是什么？",  # 若微调数据包含此策略，模型会输出正确结果
    "CVE-2099-99999属于什么漏洞类型？",      # 若微调数据有类似CVE案例，模型会类比输出
    "什么是SQL注入漏洞？如何防范？"           # 安全领域常见问题，微调后模型会更精准
]

# 批量生成并打印结果
for q in test_questions:
    ans = generate_answer(q, max_new_tokens=100)  # 生成100个token（足够回答大部分问题）
    print(f"问题：{q}")
    print(f"答案：{ans}")
    print("-" * 80)







# # --------------------------
# # 合并LoRA权重到基础模型（需足够显存）
# merged_model = fine_tuned_model.merge_and_unload()

# # 保存合并后的完整模型
# merged_model.save_pretrained("merged_secgpt_model")
# tokenizer.save_pretrained("merged_secgpt_model")

# # 后续调用时，直接加载合并后的模型（无需再加载LoRA）
# merged_model = AutoModelForCausalLM.from_pretrained(
#     "merged_secgpt_model",
#     load_in_4bit=True,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )