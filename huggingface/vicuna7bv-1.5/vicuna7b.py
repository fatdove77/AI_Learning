# Use a pipeline as a high-level helper
from transformers import pipeline
import os
pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")


# 获取当前文件所在目录
current_directory = os.path.dirname(os.path.realpath(__file__))

# 组合模型参数文件的路径
model_path = os.path.join(current_directory, "model")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5",output_dir=model_path)


# 输入文本
text = "这是一个示例输入文本。"

# 使用tokenizer将文本转换为模型输入格式
inputs = tokenizer(text, return_tensors="pt")

# 使用模型进行推理
outputs = model(**inputs)

# 获取分类结果
logits = outputs.logits