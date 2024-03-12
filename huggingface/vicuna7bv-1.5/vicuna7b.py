
import os
#
# # 设置缓存目录到 D 盘
# cache_dir = "D:/transformers_cache/"
# os.environ["TRANSFORMERS_CACHE"] = cache_dir

# # 确保缓存目录存在
# os.makedirs(cache_dir, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
input_text = "Hello, I am a language model,"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)