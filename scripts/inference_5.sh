# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --max_tokens 1 \
#     --load_bits 16 \
#     --prefix_file_name streaming_100 \
#     -m 4

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \
#     --load_bits 16 \
#     --streaming_num 5 \
#     -m 4    


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 4 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  testing \
#     -m 7    




## f32, all position and every step, wo deatch, lr_schedule
python main_modify.py \
    --input "The quick brown fox jumps over the lazy " \
    --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch_wodetatch \
    --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
    --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
    --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
    -r --record_time \
    --gamma 4 \
    --use_apt_param \
    --top_p 0 --top_k 0 \
    -s 123 \
    --fn_name sp_dy_gamma_etp_grad \
    --max_tokens 500 \
    --load_bits 16 \
    --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch_wodetatch \
    -m 0   
