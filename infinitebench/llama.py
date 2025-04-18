import json
from pathlib import Path
import time
from typing import List, Tuple, Any
import sys
import os
import torch
from torch import Tensor
from transformers import AutoTokenizer, LlamaConfig, GenerationConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)

from args import parse_args


MAX_POSITION_ID = 128*1024  # Determined by the model
TRUNCATE_LEN = 128*1024

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


# def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
#     tokens = tok.encode(input)
#     len_before = len(tokens)
#     print(f"# tokens before: {len_before}")
#     tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
#     len_after = len(tokens)  # type: ignore
#     print(f"# tokens after: {len_after}")
#     assert len_after <= len_before
#     assert len_after <= max_tokens
#     return tok.decode(tokens, skip_special_tokens=True)

def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens

def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_tokens = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")

    input_tensors = {
        "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
    }

    outputs = model.generate(**input_tensors, generation_config=generation_config)
    # outputs = model.generate(**input_tensors, generation_config=generation_config, past_key_values=cache)

    output = outputs[0, len(input_tokens) :]
    output = tok.decode(output, skip_special_tokens=True)
    output = output.strip()

    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str = "gradientai/Llama-3-8B-Instruct-262k",
    ngpu=1,
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    config = LlamaConfig.from_pretrained(model_name)
    config._attn_implementation = "flash_attention_2"
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.max_gen_len = args.max_gen_len
    config.budget = args.budget
    config.recent_size = args.recent_size
    config.start_size = args.start_size
    config.num_channel = args.num_channel
    llm = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore


if __name__ == "__main__":
    
    args = parse_args()
    model_name = args.model_name

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_name)
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        do_sample=False,
        # temperature=0,
        # top_p=0.95,
        pad_token_id=tok.pad_token_id,
    )
    # Data
    result_dir = Path(args.output_dir, model_name)
    print("result_dir: ", result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose,generation_config=generation_config
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
