from argparse import ArgumentParser, Namespace
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="./results_infinite", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--k_bits", type=int, default=2)
    p.add_argument("--v_bits", type=int, default=2)
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--residual_length", type=int, default=128)
    p.add_argument("--max_gen_len", type=int, default=1300)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--recent_size", type=int, default=64)
    p.add_argument("--start_size", type=int, default=64)
    p.add_argument("--num_channel", type=int, default=12)
    
    return p.parse_args()