from modelscope import snapshot_download


# modelscope download --model clouditera/SecGPT-1.5B --local_dir /path/to/your/target/folder

#模型下载
model_dir = snapshot_download(
    'clouditera/SecGPT-1.5B',
    local_dir='models/clouditera_SecGPT-1.5B',
)