# pip list
## 基础库
pip install torch transformers datasets accelerate

## LoRA 相关库
pip install peft bitsandbytes  # peft用于LoRA实现，bitsandbytes用于量化加载模型


# download model
models/model_download.py

# start llm
bash'''
python -m vllm.entrypoints.openai.api_server \
  --model ./models/clouditera_SecGPT-1.5B \
  --tokenizer ./models/clouditera_SecGPT-1.5B \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16
'''
## 查看运行的进程
ps aux | grep "vllm.entrypoints.openai.api_server"
ps aux | grep "python -m vllm"

# test model
test_llm.py