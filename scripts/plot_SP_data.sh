#!/bin/bash


# ##plot inference time
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
#     sp_time_sp_68m_topkp0_.csv \
#     sp_dy_gamma_time_dy_68m_topkp0.csv \
#     sp_dy_gamma_time_dy_68m_topkp0_M3P3.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_th2.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_th3.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_th4.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_th5.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_dyth3.csv \
#     sp_dy_gamma_etp_time_sp_dy_gamma_68m_topkp0_entropy_dyetp_dyth4.csv \
#     --plot_mode inference_time



# ##plot inference time
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
#     Org_SP.csv \
#     sp_time_sp_68m_topkp0__fp32.csv \
#     sp_dy_gamma_etp_time_sp_68m_topkp0_etp_golden_g4.csv \
#     sp_time_sp_68m_topkp0_fp32_500tok.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud.csv \
#     --plot_mode inference_time



# Function to set mid_names based on plot_mode
set_mid_names() {
    local plot_mode=$1
    if [ "$plot_mode" == "inference_time" ]; then
        echo "time"
    elif [ "$plot_mode" == "accepted_rate" ]; then
        echo "accepted_rate"
    else
        echo "Invalid plot mode. Please use 'inference_time' or 'accepted_rate'."
        exit 1
    fi
}

# Function to generate file names
generate_file_names() {
    local prefix_file_names=("${!1}")
    local mid_name="$2"
    local postfix_file_names=("${!3}")
    local result=()

    for i in "${!prefix_file_names[@]}"; do
        result+=("${prefix_file_names[$i]}_${mid_name}_${postfix_file_names[$i]}.csv")
    done

    echo "${result[@]}"
}

# Arrays containing prefix file names and postfix file names
prefix_file_names=(
    sp \
    sp \
    # sp_dy_gamma_etp \
    # sp_dy_gamma_etp_grad \
    # sp_dy_gamma_etp_grad \
    # sp_dy_gamma_etp_grad \
    # sp_dy_gamma_etp_grad \
    sp_dy_gamma_etp_grad \
    sp_dy_gamma_etp_grad 
)

postfix_file_names=(
    sp_68m_topkp0_fp32_500tok \
    sp_68m_topkp0_fp32_500tok_golden_g4 \
    # sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4 \
    # sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch_wodetatch \
    # sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch \
    # sp_dy_gamma_68m_topkp0_fp32_500tok_allPos \
    # sp_dy_gamma_68m_topkp0_fp32_500tok_nPos \
    sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s1Loss20Ud \
    sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud
)

# Set plot mode and get the mid_name
# plot_mode=inference_time
plot_mode=accepted_rate


mid_name=$(set_mid_names "$plot_mode")

# Call the function with the arrays and store the result in an array
file_names=($(generate_file_names prefix_file_names[@] "$mid_name" postfix_file_names[@]))

# Assign the array elements to variables p1, p2, and p3
p1=${file_names[0]}
p2=${file_names[1]}
p3=${file_names[2]}
p4=${file_names[3]}

# Plot inference time
python scripts/plot_SP_data.py \
    /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
    $p1 \
    $p2 \
    $p3 \
    $p4 \
    --plot_mode $plot_mode



# ##plot inference time
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
#     sp_time_sp_68m_topkp0_fp32_500tok.csv \
#     sp_time_sp_68m_topkp0_fp32_500tok_golden_g4.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch_wodetatch.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud.csv \
#     sp_dy_gamma_etp_grad_time_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud.csv \
#     --plot_mode inference_time




# ##plot accetped rate
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
#     sp_accepted_rate_sp_68m_topkp0_fp32_500tok.csv \
#     sp_dy_gamma_etp_accepted_rate_sp_68m_topkp0_etp_golden_g4.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch_wodetatch.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud.csv \
#     --plot_mode accepted_rate



# ##plot accetped rate
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/compare_temp \
#     sp_accepted_rate_sp_68m_topkp0__fp32.csv \
#     sp_dy_gamma_etp_accepted_rate_sp_68m_topkp0_etp_golden_g4.csv \
#     sp_accepted_rate_sp_68m_topkp0_fp32_500tok.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud.csv \
#     sp_dy_gamma_etp_grad_accepted_rate_sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud.csv \
#     --plot_mode accepted_rate



# #plot accepted_number
# python scripts/plot_SP_data.py  \
#     /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m \
#     sp_dy_gamma_kl_div_68m.csv \
#     sp_dy_gamma_accepted_sequence_68m.csv \
#     --data_number 0 \
#     --plot_mode accepted_number \
#     --suffix Q0 \
#     # --suffix KlShift_Q0 \
#     # --use_kl_shift \


# ##plot Heapmap
# python scripts/plot_SP_data.py /work/valex1377/LLMSpeculativeSampling/experiments \
#                                     speculative_la_accepted_rate_68m_g4_200to200_1000GenWithEos_.csv \
#                                     --plot_mode heatmap

     


# # #KL and accepted rate correclation
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_RejectEntropy \
#     sp_dy_gamma_reject_entropy_sp_dy_gamma_68m_topkp0_entropy_RejectEntropy.csv \
#     sp_dy_gamma_accepted_sequence_sp_dy_gamma_68m_topkp0_entropy_RejectEntropy.csv \
#     --plot_mode corr_kl_accepted \
#     # --use_kl_shift \
#     # --suffix KlShift



# ##plot Entropy time
# python scripts/plot_SP_data.py \
#     /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_entropy \
#     sp_dy_gamma_accepted_entropy_sp_dy_gamma_68m_topkp0_entropy.csv \
#     sp_dy_gamma_reject_entropy_sp_dy_gamma_68m_topkp0_entropy.csv \
#     --plot_mode entropy
