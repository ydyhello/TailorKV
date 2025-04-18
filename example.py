import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import time
from transformers import AutoTokenizer, LlamaConfig
from models.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
random.seed(100)
torch.manual_seed(100)

model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
config = LlamaConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"
config.k_bits = 1
config.v_bits = 1
config.group_size = 64
config.residual_length = 128
config.max_gen_len = 120
config.budget = 192
config.recent_size = 56
config.start_size = 8
config.num_channel = 8

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('gsm8k', 'main')

prompt=''
for i in range(5): 
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"

inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

print('-----',inputs.shape[1])
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    outputs = model.generate(inputs, use_cache=True, max_new_tokens=96)
    torch.cuda.synchronize()
    print(f'used time: {(time.time() - st) * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')

print(tokenizer.decode(outputs[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))