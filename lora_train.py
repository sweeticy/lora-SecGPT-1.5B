from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# 1. 加载模型和分词器
model_path = "models/clouditera_SecGPT-1.5B"
# 量化加载模型（节省显存，适合消费级GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,  # 4bit量化:将模型权重从默认的 32 位或 16 位精度压缩到 4 位
    device_map="auto",  #模型太大无法完全放入单张 GPU 时，会自动将部分层分配到 CPU 或其他 GPU,使用device_map={"": 0}表示强制使用编号为 0 的 GPU
    torch_dtype=torch.bfloat16,   #Brain Float 16比传统的 float16 有更大的动态范围
    trust_remote_code=True # 信任远程代码
)

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)   #加载模型配套的分词器
tokenizer.pad_token = tokenizer.eos_token  # 设置填充符,用 “填充符” 将短文本补到统一长度

# 准备模型用于量化训练：使用量化时必须有
model = prepare_model_for_kbit_training(model)

# 2. 配置LoRA参数
lora_config = LoraConfig(
    r=8,  # LoRA注意力维度
    lora_alpha=32,  # 缩放参数
    target_modules=["q_proj", "v_proj"],  # 需要添加 LoRA 适配器的模型模块
    lora_dropout=0.0,#大数据集可以适当提高防止过拟合
    bias="none",#是否微调模型的偏置项："none"（不微调偏置）、"all"（微调所有偏置）、"lora_only"（只微调 LoRA 相关的偏置）
    task_type="CAUSAL_LM"# “因果语言建模”（文本生成任务）
)

# 转换模型为LoRA模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例（应远低于1%）

# 3. 加载并预处理数据集
dataset = load_dataset("json", data_files="local_dataset/custom_security_qa.jsonl")["train"]

def preprocess_function(examples):
    # 格式化输入：instruction + input + output
    prompts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        prompt = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        prompts.append(prompt)
    
    # 分词（截断和填充）
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,  # 根据模型最大长度调整
        padding="max_length",
        return_tensors="pt"
    )

# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names  # 移除原始列
)

# 4. 配置训练参数
training_args = TrainingArguments(
    output_dir="./lora_secgpt_results",  # 输出目录
    per_device_train_batch_size=2,  # 单设备batch size（根据显存调整）
    gradient_accumulation_steps=1,  # 梯度累积（模拟更大batch）
    learning_rate=2e-4,  # LoRA学习率通常比全量微调大
    num_train_epochs=20,  # 训练轮次（小数据集可设3-5）
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="tensorboard",
    fp16=True,  # 混合精度训练（需GPU支持）
    optim="paged_adamw_8bit"  # 8bit优化器，节省显存
)

# 5. 初始化Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("开始训练...")
trainer.train()
print("训练完成！")

# 6. 保存LoRA权重（仅保存增量部分，体积小）
model.save_pretrained("lora_secgpt_weights")
print("LoRA微调完成，权重保存至 lora_secgpt_weights")











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

