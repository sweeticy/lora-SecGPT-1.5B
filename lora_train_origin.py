# train_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# 1. 加载模型和分词器
model_path = "./models/clouditera_SecGPT-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

# 2. LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 数据预处理函数
def preprocess(example):
    prompt = f"Instruction: {example['instruction']}\nInput: \nOutput: {example['output']}"
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        'input_ids': tokenized['input_ids'][0],
        'attention_mask': tokenized['attention_mask'][0],
        'labels': tokenized['input_ids'][0].clone()
    }

# 4. 加载和处理数据
dataset = load_dataset("json", data_files={"train": "local_dataset/custom_security_qa.jsonl"})["train"]
tokenized_dataset = dataset.map(# 单样本处理，自动遍历每个样本，逐个调用 preprocess 函数
    preprocess,
    batched=False,
    remove_columns=dataset.column_names
)

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./lora_security_qa",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    remove_unused_columns=False,
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("开始训练...")
trainer.train()
print("训练完成！")

# 7. 保存模型
model.save_pretrained("./lora_security_qa")

# 8. 测试生成
print("\n开始测试...")
model.eval()

model.config.temperature = None
model.config.top_p = None
model.config.top_k = None
model.config.do_sample = False

test_questions = [
    "SecGPT-1.5B的作者是谁？",
    "本地安全策略编号LS-2024-001是什么？",
    "CVE-2099-99999属于什么漏洞类型？"
]

for question in test_questions:
    prompt = f"Instruction: {question}\nInput: \nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generation_config = {
            "max_new_tokens": 50,
            "num_return_sequences": 1,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Output:")[-1].strip() if"Output:"in response else response
    
    print(f"\n问题：{question}")
    print(f"答案：{answer}")
    print("="*50)

