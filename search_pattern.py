import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import time
from transformers import AutoTokenizer, MistralConfig, AutoModelForCausalLM
from models.modeling_llama_search import LlamaForCausalLM

from datasets import load_dataset

# mistralai/Mistral-7B-Instruct-v0.3
# NousResearch/Meta-Llama-3.1-8B-Instruct
# togethercomputer/Llama-2-7B-32K-Instruct
# 01-ai/Yi-9B-200K
# 01-ai/Yi-6B-200K
# hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
config = MistralConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"

config.max_new_tokens = 96

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(50):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"

inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

print('-----',inputs.shape[1])
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    outputs = model.generate(inputs, use_cache=True, max_new_tokens=config.max_new_tokens)
    torch.cuda.synchronize()
    print(f'used time: {(time.time() - st) * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')

print(tokenizer.decode(outputs[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))