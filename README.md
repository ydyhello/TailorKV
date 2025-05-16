# TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization


- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Example](#example)
  - [Offline Identification](#offline-identification)
  - [Evaluation](#evaluation)
    - [LongBench](#longbench)
    - [RULER](#ruler)
    - [InfiniteBench](#infinitebench)
- [Acknowledgements](#acknowledgements)

# Quick Start

## Setup

To install the required packages:
```
conda create -n tailorkv python=3.12
conda activate tailorkv

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.6.3 --no-build-isolation
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
pip install transformers==4.46.1

pip install requirements.txt  

cd quant && pip install -e .
```

## Example

Load model with TailorKV: (e.g., Llama-3.1-8B-Instruct)

```python
import torch
from transformers import AutoTokenizer, LlamaConfig
from models.modeling_llama import LlamaForCausalLM

model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
config = LlamaConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"
config.k_bits = 1
config.v_bits = 1
config.group_size = 64
config.residual_length = 128
config.max_gen_len = 1300
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

# Inference
# e.g., model.generate(...)
```

We use GSM8K as an example to show how to use TailorKV.

```
python example.py
```
## Offline Identification

You can use the following script to identify the optimal compression strategy for each transformer layer:

```
python search_pattern.py
```

This will help you determine which layers are quantization-friendly and which are sparsity-friendly.

Next, specify the quantization-friendly layers in the configuration file `config/model2quantlayer.json`.

## Evaluation

### LongBench

Run LongBench with TailorKV: 

```
bash ./scripts/long_test.sh
python eval_long_bench.py --model {MODEL} # MODEL is the dir name under pred/
```

### RULER

To run the RULER benchmark, you need first download required data files:

```
cd ruler/data/synthetic/json/

# download Paul Graham Essays for the needle test
python download_paulgraham_essay.py

# download SQuAD and HotpotQA for the QA test
bash download_qa_dataset.sh

# you will need nltk.download('punkt') as well
python -c "import nltk; nltk.download('punkt')"
```

Run RULER with TailorKV: 

```
bash ./ruler/run.sh
```

### InfiniteBench

To run the InfiniteBench benchmark, you need first download required data files:

```
bash ./infinitebench/download_dataset.sh
```

Run InfiniteBench with TailorKV: (e.g., Llama-3.1-8B-Instruct)

```
bash ./infinitebench/run_llama3.sh
python eval_long_bench.py --task {task} --output_dir {output_dir} --model_name {model_name}
```

Run InfiniteBench with TailorKV: (e.g., Yi-9B-200k)

```
bash ./infinitebench/run_yi.sh
python eval_long_bench.py --task {task} --output_dir {output_dir} --model_name {model_name}
```

# Acknowledgements

We sincerely thank the Xiaomi Large Model PLUS team for their generous support in the development of TailorKV, without which this repository would not have been possible.
