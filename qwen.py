from modelscope import snapshot_download
# 以 Qwen3-8B 为例，下载后它会保存在你电脑的本地路径中
model_dir = snapshot_download('Qwen/Qwen3-8B')
print(model_dir)  # 复制打印出来的这个本地路径