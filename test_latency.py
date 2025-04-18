import os
import time
import numpy as np
from transformers import AutoTokenizer, LlamaConfig
from models.modeling_llama import LlamaForCausalLM
import pandas as pd
import tqdm
import torch
from loguru import logger
from datasets import load_dataset


def main():
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    config = LlamaConfig.from_pretrained(model_name)
    config._attn_implementation = "flash_attention_2"
    config.k_bits = 1
    config.v_bits = 1
    config.group_size = 64
    config.residual_length = 128
    config.max_gen_len = 50
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("We are loading model", model_name)

    dataset = load_dataset('gsm8k', 'main')
    input_string = ''
    for i in range(90):
        input_string += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
    input_string += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
    
    repeat_prompt = ",".join([input_string for _ in range(10)])
    prompt = f"[INST]{repeat_prompt}[/INST]"
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    print("repeat_prompt",len(input_[0]))
    gen_max_token = 20
    # 第一遍要预热，让gpu把kernel都加载上
    for idx in range(5):
        for seqlen in tqdm.tqdm([16000,32000,64000,96000]):
            print("---------",input_.input_ids[:, :seqlen].shape[1])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=1, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]} \r")
            torch.cuda.synchronize()
            end = time.perf_counter()
            ttft = end - begin
            
            torch.cuda.synchronize()
            time.sleep(30)
            
            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=2, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]} \r")
            torch.cuda.synchronize()
            end = time.perf_counter()
            tt2t = end - begin
            
            torch.cuda.synchronize()
            time.sleep(30)

            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=gen_max_token, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]}")
            print("len",len(output))
            torch.cuda.synchronize()
            end = time.perf_counter()
            time.sleep(30)
            decoding_elapsed = end - begin - tt2t

            print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                    f"ttft is {ttft},"
                    f"tt2t is {tt2t},"
                    f"decoding elasped:{decoding_elapsed},"
                    f"{decoding_elapsed / (gen_max_token - 2)} per decoding token.")
    del model

    logger.info(f"del objects done.")   
    exit()

if __name__ == "__main__":
    main()
