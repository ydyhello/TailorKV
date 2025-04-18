gpuid=0
model=NousResearch/Meta-Llama-3.1-8B-Instruct
method=tailorkv
k_bits=1
v_bits=1
group_size=64
residual_length=128
max_gen_len=1300
budget=192
recent_size=56
start_size=8
num_channel=8
e=0


CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --method $method \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --max_gen_len $max_gen_len \
    --budget $budget \
    --recent_size $recent_size \
    --start_size $start_size \
    --num_channel $num_channel \
    --e ${e}