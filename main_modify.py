
import torch
import argparse
import contexttimer
from colorama import Fore, Style
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import ( 
                      autoregressive_sampling, speculative_sampling, 
                      speculative_sampling_v2, speculative_sampling_google,
                      autoregressive_sampling_with_eos, speculative_sampling_google_streaming,
                      self_speculative_sampling_google_streaming,
                      self_streaming,
                      test_time,
                      speculative_sampling_google_local_atten,
                      speculative_sampling_google_local_input,
                      speculative_sampling_google_dynamic_gamma,
                      speculative_sampling_google_dynamic_gamma_entropy,
                      speculative_sampling_google_dynamic_gamma_entropy_grad,
)
from globals import Decoder
from datasets import mt_bench_dataset
from transformers import (
    default_data_collator,
)
import os
from tqdm import tqdm
import subprocess
import numpy as np


# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}
            
TESTFUNSET = {
    "at" : autoregressive_sampling_with_eos,
    "sp" : speculative_sampling_google,
    "sp_streaming" : self_speculative_sampling_google_streaming,
    "sp_la" : speculative_sampling_google_local_atten,
    "sp_li" : speculative_sampling_google_local_input,
    "sp_dy_gamma" : speculative_sampling_google_dynamic_gamma ,
    "sp_dy_gamma_etp" : speculative_sampling_google_dynamic_gamma_entropy,
    "sp_dy_gamma_etp_grad" :speculative_sampling_google_dynamic_gamma_entropy_grad,
    
}        




def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama2-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama2-70b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--record_accepted_count', '-r', action='store_true', default=False, help='recode accpeted number results.') 
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--dataset_path', '-dp', type=str, default=None, help='dataset path')
    parser.add_argument('--golden_accepted_sequence_path', '-gasp', type=str, default=None, help='golden accepted sequence path')
    parser.add_argument('--result_folder', '-rf', type=str, default=None, help='test')
    parser.add_argument('--load_bits', '-lb', type=int, default=16, help='4 / 8 / 16 for approx model bits ')
    parser.add_argument('--mode', '-m', type=int, default=0, help='0 / 1 / 2/3 for approx model bits ')
    parser.add_argument('--batch_mode', type=int, default=0, help='0 / 1 / 2/ 3 for AT/SP/SSP ')
    parser.add_argument('--streaming_num', type=int, default=10, help=' ')
    parser.add_argument('--prefix_file_name', type=str, default="_")
    parser.add_argument('--fn_name', type=str, default=TESTFUNSET["sp"])    
    parser.add_argument('--file_root', type=str, default='/work/valex1377/LLMSpeculativeSampling/scripts')   
    parser.add_argument('--record_time', action='store_true', default=False, help='save average excution time.') 
    parser.add_argument('--top_k', type=int, default=0, help='top k sample in approx model')
    parser.add_argument('--top_p', type=int, default=0, help='top p sample in approx model')  
    parser.add_argument('--use_apt_param', action='store_true', default=False, help='the parameter of approx model is change by kl_loss')
      
    
    args = parser.parse_args()
    return args



def read_csv_to_list(file_path):
    import csv
    data_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each row into a list of integers
            row = [int(item) for item in row if item]  # Skip empty items
            data_list.append(row)
    return data_list



def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def get_gpu_usage():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage, memory_usage = map(int, result.strip().split(', '))
    return gpu_usage, memory_usage    

def accepted_count_benchmark(fn, file_path, *args, **kwargs):
    output,_,_,_,accepted_number = fn( *args, **kwargs)
    accepted_number = np.array(accepted_number)
    accepted_number = accepted_number.reshape(1, -1)
    sum_of_accepted_numbers = np.sum(accepted_number)   
    print("output length:{} , sum_of_accepted_numbers: {}".format(len(output[0]),sum_of_accepted_numbers))
    with open(file_path, 'ab') as f:
        np.savetxt(f, accepted_number, fmt='%s', delimiter=',')        
        
# def accepted_count_sum_benchmark(fn, *args, **kwargs):
#     output,_,_,_,accepted_number = fn( *args, **kwargs)
#     accepted_number = np.array(accepted_number)
#     accepted_number = accepted_number.reshape(1, -1)
#     # Compute the sum of accepted numbers
#     sum_of_accepted_numbers = np.sum(accepted_number)   
#     accepted_rate = sum_of_accepted_numbers/len(output[0])
#     print("output length:{} , sum_of_accepted_numbers: {} , Accpet Rate:{} ".format(len(output[0]),sum_of_accepted_numbers,accepted_rate))
#     return sum_of_accepted_numbers, accepted_rate, len(output[0]) , accepted_number
          
          
          
def save_data(file_root, mid_name, prefix_file_name, record_time, sum_of_accepted_numbers, 
                         accepted_rate, output_length, accepted_sequence, average_time,
                         kl_div,gamma_sequence, accepted_entropy,reject_entropy, generate_tokens):

    # Extract data from the tuple
    
    accepted_numbers_file_path=f'{file_root}/{mid_name}_accepted_number_{prefix_file_name}.csv'
    accepted_rate_file_path=f'{file_root}/{mid_name}_accepted_rate_{prefix_file_name}.csv'
    output_length_file_path=f'{file_root}/{mid_name}_output_length_{prefix_file_name}.csv'
    accepted_sequence_file_path=f'{file_root}/{mid_name}_accepted_sequence_{prefix_file_name}.csv'
    time_file_path=f'{file_root}/{mid_name}_time_{prefix_file_name}.csv'
    generate_tokens_path=f'{file_root}/{mid_name}_generated_{prefix_file_name}.csv'
    kl_div_path=f'{file_root}/{mid_name}_kl_div_{prefix_file_name}.csv'
    gamma_sequence_path=f'{file_root}/{mid_name}_gamma_sequence_{prefix_file_name}.csv'
    accepted_entropy_path=f'{file_root}/{mid_name}_accepted_entropy_{prefix_file_name}.csv'
    reject_entropy_path=f'{file_root}/{mid_name}_reject_entropy_{prefix_file_name}.csv'
    
    sum_of_accepted_numbers = np.array(sum_of_accepted_numbers)
    sum_of_accepted_numbers = sum_of_accepted_numbers.reshape(1, -1)
    accepted_rate = np.array(accepted_rate)
    accepted_rate = accepted_rate.reshape(1, -1)             
    output_length = np.array(output_length)
    output_length = output_length.reshape(1, -1)                         
    with open(accepted_numbers_file_path, 'ab') as f:
        np.savetxt(f, sum_of_accepted_numbers, fmt='%s', delimiter=',')       
    with open(accepted_rate_file_path, 'ab') as f:
        np.savetxt(f, accepted_rate, fmt='%s', delimiter=',')             
    with open(output_length_file_path, 'ab') as f:
        np.savetxt(f, output_length, fmt='%s', delimiter=',')  
    with open(accepted_sequence_file_path, 'ab') as f:
        np.savetxt(f, accepted_sequence, fmt='%s', delimiter=',')  
    with open(generate_tokens_path, 'a',encoding='utf-8') as f:
        generate_tokens = ["[_START_]"] + generate_tokens
        np.savetxt(f, generate_tokens, fmt='%s', delimiter=',')    
    if kl_div is not None:
        with open(kl_div_path, 'ab') as f:
            np.savetxt(f, kl_div, fmt='%s', delimiter=',')           
    if accepted_entropy is not None:
        with open(accepted_entropy_path, 'ab') as f:
            np.savetxt(f, accepted_entropy, fmt='%s', delimiter=',')              
    if reject_entropy is not None:
        with open(reject_entropy_path, 'ab') as f:
            np.savetxt(f, reject_entropy, fmt='%s', delimiter=',')                
    if gamma_sequence is not None:
        with open(gamma_sequence_path, 'ab') as f:
            np.savetxt(f, gamma_sequence, fmt='%s', delimiter=',')                    
    if record_time :
        with open(time_file_path, 'ab') as f:
            np.savetxt(f, average_time, fmt='%s', delimiter=',')            
        
def save_time_data(time_list, file_root , mid_name, prefix_file_name):
    time_file_path=f'{file_root}/{mid_name}_time_{prefix_file_name}.csv'
    time_list = np.array(time_list)
    time_list = time_list.reshape(1, -1)
    with open(time_file_path, 'ab') as f:
        np.savetxt(f, time_list, fmt='%s', delimiter=',')               



def process_kwargs(**kwargs):
    # Ensure 'golden_accepted_sequence' is a list
    if kwargs['golden_accepted_sequence'] is None:
        kwargs.pop('golden_accepted_sequence')
    return kwargs

          
def recoder_benchmark(fn, file_root, mid_name, prefix_file_name, TEST_TIME=1, record_time = False , *args, **kwargs):
    
    
    kwargs = process_kwargs(**kwargs) 
    
    if record_time == False: 
        TEST_TIME = 1
        
    with contexttimer.Timer() as t:
        for _ in range(TEST_TIME): 
            fn_outputs = fn( *args, **kwargs)
    
    if fn_outputs.output_tokens.dim() == 2 :
        length_output_tokens = len(fn_outputs.output_tokens[0])
    else:
        length_output_tokens = len(fn_outputs.output_tokens)
    average_time = np.array([length_output_tokens / (t.elapsed / TEST_TIME)])
    
    accepted_number = np.array(fn_outputs.accepted_num)
    accepted_number = accepted_number.reshape(1, -1)
    # Compute the sum of accepted numbers
    sum_of_accepted_numbers = np.sum(accepted_number)   
    accepted_rate = sum_of_accepted_numbers/length_output_tokens
    
    kl_div_out = getattr(fn_outputs, 'kl_div_out', None)
    if kl_div_out is not None :
        kl_div_out = np.array(kl_div_out)
        kl_div_out = kl_div_out.reshape(1, -1)        

    accepted_entropy = getattr(fn_outputs, 'accepted_entropy', None)
    if accepted_entropy is not None :
        accepted_entropy = np.array(accepted_entropy)
        accepted_entropy = accepted_entropy.reshape(1, -1)   
        
    reject_entropy = getattr(fn_outputs, 'reject_entropy', None)
    if reject_entropy is not None :
        reject_entropy = np.array(reject_entropy)
        reject_entropy = reject_entropy.reshape(1, -1)                   
    
    gamma_sequence = getattr(fn_outputs, 'gamma_sequence', None)
    if gamma_sequence is not None :
        gamma_sequence = np.array(gamma_sequence)
        gamma_sequence = gamma_sequence.reshape(1, -1)  

        
        
    
    
    
    print(
            f"output length:{length_output_tokens} , tokens/sec: {length_output_tokens / (t.elapsed / TEST_TIME)},\
            {t.elapsed / TEST_TIME} sec generates {length_output_tokens} tokens , sum_of_accepted_numbers: {sum_of_accepted_numbers},\
            Accpet Rate:{accepted_rate}"
    )
    # print("output length:{} , sum_of_accepted_numbers: {} , Accpet Rate:{} ".format(length_output_tokens,sum_of_accepted_numbers,accepted_rate))
    output_words = Decoder().decode(fn_outputs.output_tokens)
    print(f"\033[31m{output_words}\033[0m")    
    
    save_data(file_root, mid_name, prefix_file_name, record_time, 
              sum_of_accepted_numbers, accepted_rate, length_output_tokens , accepted_number, average_time, 
              kl_div_out, gamma_sequence, accepted_entropy,reject_entropy,[output_words])

          
def benchmark_sum(fn, print_prefix, use_profiler=True,TEST_TIME:int=10, *args, **kwargs):
    profile_filename = f"./profile_logs/{print_prefix}"
    print("============START MEASURE TEST==========")
    gpu_usage=[] ; memory_usage=[]
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)
        
    print(f"\n output length:{len(output[0])} , tokens/sec: {len(output[0]) / (t.elapsed / TEST_TIME)}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")
    print(f"\033[31m{Decoder().decode(output)}\033[0m")
    return np.array([len(output[0]) / (t.elapsed / TEST_TIME)])

  
        
    

def benchmark(fn, print_prefix, use_profiler=True,file_name=None,TEST_TIME:int=10, *args, **kwargs):
    profile_filename = f"./profile_logs/{print_prefix}"
    print("============START MEASURE TEST==========")
    gpu_usage=[] ; memory_usage=[]
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)
    
    
    # gpu,memory=get_gpu_usage()
    # gpu_usage.append(gpu)
    # memory_usage.append(memory)               
    # avg_gpu = np.array([sum(gpu_usage)/TEST_TIME])
    # avg_memory = np.array([sum(memory_usage)/TEST_TIME])    
    # print("avg_gpu = {}".format(sum(gpu_usage)/TEST_TIME))
    # print("avg_memory = {}".format(sum(memory_usage)/TEST_TIME))
    # with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/avg_gpu_{file_name}.csv', 'ab') as f:
    #     np.savetxt(f, avg_gpu, fmt='%s')     
    # with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/avg_memory_{file_name}.csv', 'ab') as f:
    #     np.savetxt(f, avg_memory, fmt='%s')     

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / (t.elapsed / TEST_TIME)}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")
    print(f"\033[31m{Decoder().decode(output)}\033[0m")
    verdicts = np.array([len(output[0]) / (t.elapsed / TEST_TIME)])

    if file_name is None :
        file_name=''
    with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/time_{file_name}.csv', 'ab') as f:
        np.savetxt(f, verdicts, fmt='%s')
             
        

    
def generate(input_text,  approx_model_name, target_model_name, small_model, large_model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)

    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)



def test_speculative_time(input_text, approx_model_name, target_model_name, small_model, large_model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,load_bits=16):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    


    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9
    TEST_TIME = 10

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_large", use_profiling,
    #               input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_small", use_profiling,
    #               input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
 
 
    # torch.manual_seed(123)
    # num_tokens = 500
    # eos_token_id = tokenizer.eos_token_id ##
    # output,accepted_count,target_sample_count,resample_count,accepted_number = speculative_sampling_google(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, eos_token_id=eos_token_id, output_count=True)
    # accepted_number = np.array(accepted_number)
    # accepted_number = accepted_number.reshape(1, -1)
    # with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/speculative_accepted_number_{prefix_file_name}.csv', 'ab') as f:
    #     np.savetxt(f, accepted_number, fmt='%s', delimiter=',') 
    
    
    if use_benchmark:       
        for num_tokens in range(10,3000,100) :    
            # output = speculative_sampling_google(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # color_print(f"google's speculative_sampling: {generated_text}")
            print(f"=================={num_tokens}=================")  
            file_name=f'{prefix_file_name}'
            benchmark(speculative_sampling_google, "SP", use_profiling,file_name , TEST_TIME,
                    input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma , top_k = top_k, top_p=top_p, random_seed = random_seed)

def test_at_time(input_text,  approx_model_name, target_model_name, small_model, large_model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,load_bits=16):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
  
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    # torch.manual_seed(123)
    # output = autoregressive_sampling_with_eos(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p, eos_token_id = tokenizer.eos_token_id)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    
    # if use_benchmark:
    #     print("   ======={}======".format(num_tokens))
    #     benchmark(autoregressive_sampling_with_eos, "AS_small", use_profiling,
    #             input_ids, small_model, N = num_tokens, top_k = top_k, top_p=top_p, eos_token_id = tokenizer.eos_token_id)    
    # for i in range(1000) :    
    #     if use_benchmark:
    #         print("   ======={}======".format(i))
    #         benchmark(autoregressive_sampling, "AS_small", use_profiling,
    #                 input_ids, small_model, N = i, top_k = top_k, top_p=top_p)
    TEST_TIME = 10
    if use_benchmark:       
        for num_tokens in range(10,3000,100) :    
            # output = speculative_sampling_google(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # color_print(f"google's speculative_sampling: {generated_text}")
            file_name=f'{prefix_file_name}'
            print(f"=================={num_tokens}=================")  
            benchmark(autoregressive_sampling, "AT", use_profiling,file_name , TEST_TIME,
                    input_ids, large_model, max_len = num_tokens, top_k = top_k, top_p=top_p, 
                    )    

def save_acceptance_data(file_root, mid_name, prefix_file_name,sum_of_accepted_numbers, 
                         accepted_rate, output_length, accepted_sequence ):

    # Extract data from the tuple
    
    accepted_numbers_file_path=f'{file_root}/{mid_name}_accepted_number_{prefix_file_name}.csv'
    accepted_rate_file_path=f'{file_root}/{mid_name}_accepted_rate_{prefix_file_name}.csv'
    output_length_file_path=f'{file_root}/{mid_name}_output_length_{prefix_file_name}.csv'
    accepted_sequence_file_path=f'{file_root}/{mid_name}_accepted_sequence_{prefix_file_name}.csv'
    
    sum_of_accepted_numbers = np.array(sum_of_accepted_numbers)
    sum_of_accepted_numbers = sum_of_accepted_numbers.reshape(1, -1)
    accepted_rate = np.array(accepted_rate)
    accepted_rate = accepted_rate.reshape(1, -1)             
    output_length = np.array(output_length)
    output_length = output_length.reshape(1, -1)                         
    with open(accepted_numbers_file_path, 'ab') as f:
        np.savetxt(f, sum_of_accepted_numbers, fmt='%s', delimiter=',')       
    with open(accepted_rate_file_path, 'ab') as f:
        np.savetxt(f, accepted_rate, fmt='%s', delimiter=',')             
    with open(output_length_file_path, 'ab') as f:
        np.savetxt(f, output_length, fmt='%s', delimiter=',')  
    with open(accepted_sequence_file_path, 'ab') as f:
        np.savetxt(f, accepted_sequence, fmt='%s', delimiter=',')  
def save_time_data(time_list, file_root , mid_name, prefix_file_name):
    time_file_path=f'{file_root}/{mid_name}_time_{prefix_file_name}.csv'
    time_list = np.array(time_list)
    time_list = time_list.reshape(1, -1)
    with open(time_file_path, 'ab') as f:
        np.savetxt(f, time_list, fmt='%s', delimiter=',')       





def generate_batch(file_root, fn_name,  approx_model_name, target_model_name, small_model, large_model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, record_accepted_count=False, use_profiling = False, dataset_path=None,
             golden_accepted_sequences = None, top_k=0,top_p=0,result_folder="mt_bench",streaming_num = 10, record_time = False):

    # top_k = 20
    # top_p = 0.9
    batch_size = 1 
    torch.manual_seed(123)
    
    TEST_TIME = 3
    
        
    if batch_size > 1 :
        ValueError("not support the batchsize > 1")
    num_workers_dataloader = 4
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)    
    print(f"begin loading models - approx_model_name/target_model_name: \n {approx_model_name} \n {target_model_name}")    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
    eos_token_id = tokenizer.eos_token_id
    dataset_test = mt_bench_dataset(dataset_path, tokenizer)
    test_sampler = None
    # Create DataLoaders for the testing and validation dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers_dataloader,
        pin_memory=True,
        sampler=test_sampler if test_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )       
    

    # with open(output_path, "w", buffering=1, encoding="utf-8") as result_path:       
    for step, batch in enumerate(tqdm(test_dataloader,colour="blue", desc=f"Testing num: ", total=len(test_dataloader), dynamic_ncols=True)):      
        if batch['labels'] is not None:
            batch_labels = batch['labels']
            batch.pop("labels",None)            
        for key in batch.keys():
            batch[key] = batch[key].to(torch_device) 
      
        print(f"==================The {step} data =================")  
        
        fn = TESTFUNSET[fn_name]
        mid_name = fn_name
        
        golden_accepted_sequence = golden_accepted_sequences[step] if golden_accepted_sequences is not None else None
        
        
        recoder_benchmark(fn, file_root, mid_name, prefix_file_name, TEST_TIME, record_time , batch['input_ids'], 
                        small_model, large_model, max_len = num_tokens, gamma = gamma ,top_k = top_k, top_p=top_p, 
                        random_seed = random_seed, eos_token_id=eos_token_id, output_count=True, golden_accepted_sequence=golden_accepted_sequence)
            
        




def test_speculative_streaming_time(input_text, approx_model_name, target_model_name, small_model, large_model, num_tokens=20, gamma = 4,prefix_file_name="",
                                    
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,streaming_num = 10):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    


    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_large", use_profiling,
    #               input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_small", use_profiling,
    #               input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
 
   
    torch.manual_seed(123)
    num_tokens = 1000
    eos_token_id = tokenizer.eos_token_id ##
    for streaming_num in range(100):
        output,accepted_count,target_sample_count,resample_count,accepted_number = speculative_sampling_google_streaming(input_ids, small_model, 
                                                                                    large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed,
                                                                                    eos_token_id=eos_token_id, output_count=True,streaming_num=streaming_num)
        accepted_number = np.array(accepted_number)
        accepted_number = accepted_number.reshape(1, -1)
        with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/speculative_accepted_number_{prefix_file_name}tokenAtten.csv', 'ab') as f:
            np.savetxt(f, accepted_number, fmt='%s', delimiter=',') 
    
    
    if use_benchmark:       
        for num_tokens in range(1000) :    
            # output = speculative_sampling_google(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # color_print(f"google's speculative_sampling: {generated_text}")
            file_name=f'speculative_{prefix_file_name}tokenAtten'
            print(f"=================={num_tokens}=================")  
            benchmark(speculative_sampling_google_streaming, "SP", use_profiling,file_name ,
                    input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)




def test_self_speculative_streaming_time(input_text, model_name, model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,streaming_num = 10):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)


    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_large", use_profiling,
    #               input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_small", use_profiling,
    #               input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
 
   
    torch.manual_seed(123)
    eos_token_id = tokenizer.eos_token_id ##
    # # for i in range(streaming_num,streaming_num+1):
    for i in range(2,streaming_num):
        output,accepted_count,target_sample_count,resample_count,accepted_number = self_speculative_sampling_google_streaming(input_ids, model,
                                                                                    num_tokens, gamma=gamma , top_k = top_k, top_p=top_p, random_seed = random_seed,
                                                                                    eos_token_id=eos_token_id, output_count=True,streaming_num=i)
        accepted_number = np.array(accepted_number)
        accepted_number = accepted_number.reshape(1, -1)
        with open(f'/work/valex1377/LLMSpeculativeSampling/scripts/self_speculative_accepted_number_{prefix_file_name}tokenAtten.csv', 'ab') as f:
            np.savetxt(f, accepted_number, fmt='%s', delimiter=',') 
    
    
    # if use_benchmark:       
    #     for num_tokens in range(500) :    
    #         # output = speculative_sampling_google(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    #         # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #         # color_print(f"google's speculative_sampling: {generated_text}")
    #         file_name=f'self_speculative_{prefix_file_name}tokenAtten'
    #         print(f"=================={num_tokens}=================")  
    #         benchmark(self_speculative_sampling_google_streaming, "SP", use_profiling,file_name ,
    #                 input_ids, model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed,
    #                 eos_token_id=eos_token_id, output_count=False,streaming_num_input=streaming_num)








def test_self_streaming_time(input_text, model_name, model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,streaming_num = 10):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)


    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    eos_token_id = tokenizer.eos_token_id ##
    
    if use_benchmark:       
        for num_tokens in range(500) :    
            file_name=f'self_streaming_{prefix_file_name}tokenAtten'
            print(f"=================={num_tokens}=================")  
            benchmark(self_streaming, "SP", use_profiling,file_name ,
                    input_ids, model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed,
                    eos_token_id=eos_token_id, output_count=False,streaming_num_input=streaming_num)







def test_generate_model(input_text, model_name, model, num_tokens=20, gamma = 4,prefix_file_name="",
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,streaming_num = 10):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)


    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    eos_token_id = tokenizer.eos_token_id ##
    
    output,accepted_count,target_sample_count,resample_count,accepted_number = test_time(input_ids, model,
                                                                                num_tokens, gamma=gamma , top_k = top_k, top_p=top_p, random_seed = random_seed,
                                                                                eos_token_id=eos_token_id, output_count=True,streaming_num=streaming_num)

    


def get_model(approx_model_name, target_model_name, small_load_bits : int = 16, target_load_bits : int = 16, fn_name : str=None):
    print(f"begin loading models: \n approx : {approx_model_name} \n target : {target_model_name}")
    if approx_model_name is not None :
        if small_load_bits == 4 :
            small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                            load_in_4bit=True, 
                                                            torch_dtype=torch.float16,
                                                            device_map="auto",
                                                            trust_remote_code=True)  
        elif small_load_bits == 8 :
            small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                            load_in_8bit=True, 
                                                            torch_dtype=torch.float16,
                                                            device_map="auto",
                                                            trust_remote_code=True)         
        else:
            # small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
            #                                                 torch_dtype=torch.float16,
            #                                                 device_map="auto",
            #                                                 trust_remote_code=True)    
            
            small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                            device_map="auto",
                                                            trust_remote_code=True)               
        print("finish loading small model")
    else:
        small_model = None
               
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)    
    
    if small_model is not None :
        small_model.eval()
    if large_model is not None :
        large_model.eval() 
        
    if fn_name == 'sp_dy_gamma_etp_grad':
        for param in small_model.parameters():
            param.requires_grad = True        
        small_model.train()     

    print("finish loading large_model")
    return small_model, large_model
    

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode in [2,5]:
        args.approx_model_name = None
        

    file_root = args.file_root
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    
    mid_name = args.fn_name
    golden_accepted_sequences = None
    if args.golden_accepted_sequence_path is not None:
        golden_accepted_sequences = read_csv_to_list(args.golden_accepted_sequence_path)
        

        
    small_model, large_model = get_model(args.approx_model_name, args.target_model_name, small_load_bits = args.load_bits, target_load_bits = 16, fn_name = args.fn_name)

        
            
    if args.mode == 0:
        print("BATCH MODE:{}".format(args.batch_mode))
        generate_batch(args.file_root,args.fn_name, args.approx_model_name, args.target_model_name, small_model, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, top_k = args.top_k, top_p = args.top_p,
                record_accepted_count=args.record_accepted_count, dataset_path = args.dataset_path, golden_accepted_sequences=golden_accepted_sequences, result_folder=args.result_folder,
                streaming_num = args.streaming_num, record_time = args.record_time)      
    elif args.mode == 1:
        generate(args.input, args.approx_model_name, args.target_model_name, small_model, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)        
        
    elif args.mode == 2:
        test_at_time(args.input, args.approx_model_name, args.target_model_name, small_model, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, load_bits=args.load_bits)           
        
    elif args.mode == 3:
        test_speculative_time(args.input, args.approx_model_name, args.target_model_name, small_model, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, load_bits=args.load_bits)         
        
    elif args.mode == 4:
        test_speculative_streaming_time(args.input, args.approx_model_name, args.target_model_name, small_model, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, streaming_num = args.streaming_num)          
    elif args.mode == 5:
        test_self_speculative_streaming_time(args.input, args.target_model_name, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, streaming_num = args.streaming_num)          
    elif args.mode == 6:
        test_self_streaming_time(args.input, args.target_model_name, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
            prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, streaming_num = args.streaming_num)          
      
        
    elif args.mode == 8:
        test_generate_model(args.input, args.target_model_name, large_model, num_tokens=args.max_tokens, gamma=args.gamma,
                prefix_file_name=args.prefix_file_name,random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, streaming_num = args.streaming_num)      
                
    else:
        raise ValueError
        
        
    

    
  
    
    

