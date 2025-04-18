
model_name="NousResearch/Meta-Llama-3.1-8B-Instruct"
k_bits=2
v_bits=2
group_size=64
residual_length=128
max_gen_len=1300
budget=1024
recent_size=64
start_size=64
num_channel=12

for task in number_string passkey math_find code_debug longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng; do
    CUDA_VISIBLE_DEVICES=0 python ./infinitebench/llama.py --task ${task} \
    --output_dir ./infinitebench/results \
    --model_name $model_name \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --max_gen_len $max_gen_len \
    --budget $budget \
    --recent_size $recent_size \
    --start_size $start_size \
    --num_channel $num_channel
done
