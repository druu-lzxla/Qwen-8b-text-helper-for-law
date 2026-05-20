from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# 指定你刚刚转换好的本地模型路径
model_path = "Qwen3-8B-int4-ov"

print("正在加载模型和分词器...")
# 加载模型，device 指定为 'CPU' 或 'GPU' (如果你有 Intel 核显/独显)
model = OVModelForCausalLM.from_pretrained(model_path, device='CPU')
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("模型加载成功！准备开始对话...\n")

# 构造 Qwen3 的对话模板
prompt = "你好，请用一句话介绍一下你自己。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 将文本转化为模型能看懂的输入格式
model_inputs = tokenizer([text], return_tensors="pt")

print("模型正在思考中...\n")
# 开始生成回答，max_new_tokens 限制生成的最大字数
generated_ids = model.generate(**model_inputs, max_new_tokens=512)

# 提取并解码生成的回答
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("模型回答：")
print("-" * 20)
print(response)
print("-" * 20)