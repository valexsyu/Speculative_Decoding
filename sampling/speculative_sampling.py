import torch
from tqdm import tqdm
import torch
from sampling.kvcache_model import (
    KVCacheModel, KVCacheModelStreaming, 
    KVCacheModelStreamingOneModel,test_time_Model,
    KVCacheLocalStreamingModel, 
    KVCacheModelEntropyGamma, KVCacheModelEntropyGammaGard,
)
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
import torch.nn.functional as F

import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from collections import namedtuple
SpModelOut = namedtuple(
    "SpModelOut" , 
    "output_tokens accepted_count target_sample_count resample_count accepted_num" 
)
SpDyGammaModelOut = namedtuple(
    "SpDyGammaModelOut" , 
    SpModelOut._fields + ("kl_div_out","gamma_sequence","accepted_entropy","reject_entropy",)
)

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        tot_num += 1#### 

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    return prefix


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    tot_num = 0 ####
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            tot_num += 1#### 
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :])) 
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)
    print("============================={}===================".format(tot_num))####

    return prefix

@torch.no_grad()
def speculative_sampling_google(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4, golden_accepted_sequence : list = None, 
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, eos_token_id : int = None, output_count : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    golden_accepted_sequence_temp = golden_accepted_sequence.copy() if golden_accepted_sequence is not None else None
    input_gamma = gamma
    
    
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        
        # x = approx_model_cache.generate(prefix, gamma)
        # _ = target_model_cache.generate(x, 1)
        
        if golden_accepted_sequence_temp is not None:
            assert gamma >= 0, print("gamma < 0")
            if len(golden_accepted_sequence_temp) == 0 :
                gamma = input_gamma
            else:
                gamma = golden_accepted_sequence_temp.pop(0)
            if gamma < input_gamma :
                gamma = gamma + 1

        
        if gamma > 0 :
            x = approx_model_cache.generate(prefix, gamma)
            _ = target_model_cache.generate(x, 1)
        else:
            x = prefix
            _ = target_model_cache.generate(x, 1)        
        
        n = prefix_len + gamma - 1
        tot_num += 1

        accepted_count_single_time = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
        accepted_num.extend([accepted_count_single_time])
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        if approx_model_cache._prob_history is not None:
            approx_model_cache.rollback(n+1)
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
          
                        
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
            
            
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                
                break
    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
        return SpModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                )            
    else:
        return prefix[:,seq_len:]




@torch.no_grad()
def speculative_sampling_google_streaming(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False ,streaming_num : int = 10 ) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModelStreaming(approx_model, temperature, top_k, top_p, streaming_num)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        tot_num += 1#### 

        accepted_count_single_time = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
        
        accepted_num.extend([accepted_count_single_time])
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                
                break

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
        return SpModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                )            
    else:
        return prefix[:,seq_len:]
    
    


















@torch.no_grad()
def self_speculative_sampling_google_streaming(prefix : torch.Tensor, model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False ,streaming_num : int = 10 ) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    device = model.device
    model_cache = KVCacheModelStreamingOneModel(model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    
    
    if model_cache._past_key_values is None or seq_len < streaming_num :
        if seq_len < streaming_num :
            prefix, _ = model_cache.generate(prefix, streaming_num - seq_len, approx_model_action=False,streaming_num=10000)
            target_sample_count = target_sample_count + streaming_num - seq_len
        else:
            prefix, _ = model_cache.generate(prefix, 1, approx_model_action=False,streaming_num=10000)
            target_sample_count += 1
            
            
    
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        x,approx_prob_history = model_cache.generate(prefix, gamma, approx_model_action=True, streaming_num=streaming_num)
        _,target_prob_history = model_cache.generate(x, 1, approx_model_action=False,streaming_num=10000)
        n = prefix_len + gamma - 1
        tot_num += 1#### 
        accepted_count_single_time = 0
        # print("x.size = {} , approx_prob_history.size = {} ,  target_prob_history.size = {}".format(x.size(),approx_prob_history.size(), target_prob_history.size()))
        # print("gamma = {}, prefix_len = {}".format(gamma, prefix_len))
        
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            # print("i={}".format(i))
            
            if r > (target_prob_history[:, prefix_len + i - 1, j]) / (approx_prob_history[:, streaming_num + i, j]):
            # if r > (target_prob_history[:, prefix_len + i - 1, j]) / (approx_prob_history[:, -gamma+i, j]):
                # reject
                n = prefix_len + i - 1
                approx_n =  streaming_num + i
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            

            accepted_count += 1
            accepted_count_single_time += 1
        
        accepted_num.extend([accepted_count_single_time])
        # print("token length: {:03d} . Output:{}".format(len(prefix[0, seq_len:]), Decoder().decode(prefix[:, seq_len:])))
        # print("n = {}, i = {}".format(n,i))
        # print("target_kv.size = {}".format(model_cache._past_key_values[0][0].size()))
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        
        
        # assert approx_prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_prob_history[:, n, :] - approx_prob_history[:, approx_n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_prob_history.shape[1] - 1
            t = sample(target_prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            model_cache.rollback(n+2)
        
        prefix = torch.cat((prefix, t), dim=1)
        # if eos_token_id is not None :
        #     if eos_token_id in prefix[0] : ## only support batch = 1
        #         # find the indices where x is equal to eos_token
        #         eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

        #         if eos_indices.nelement() > 0:
        #             # get the index of the first occurrence of eos_token
        #             first_eos_index = eos_indices[0]

        #             # select the elements in x before the first eos_token
        #             prefix = prefix[:,:first_eos_index]             
                
        #         break

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
        return SpModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                )            
    else:
        # print(f"target samples {n}: \033[35m{Decoder().decode(prefix[:,seq_len:])}\033[0m")
        # breakpoint()        
        return prefix[:,seq_len:]
        

@torch.no_grad()
def self_streaming(prefix : torch.Tensor, model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False ,streaming_num_input : int = 10 ) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    device = model.device
    model_cache = KVCacheModelStreamingOneModel(model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        
        if prefix_len < streaming_num_input :
            streaming_num = prefix_len
        else:
            streaming_num = streaming_num_input
        x,approx_prob_history = model_cache.generate(prefix, gamma, approx_model_action=True, streaming_num=streaming_num)
        prefix = x


    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
    else:
        return prefix[:,seq_len:]
                

                
                
                

@torch.no_grad()
def test_time(prefix : torch.Tensor, model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False ,streaming_num : int = 10 ) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    device = model.device
    model_cache = test_time_Model(model, temperature, top_k, top_p)
    x,approx_prob_history = model_cache.generate(prefix, gamma, approx_model_action=True, streaming_num=streaming_num)
    

    print(f"target samples : \033[35m{Decoder().decode(prefix[:,seq_len:])}\033[0m")
    return x[:,seq_len:]            







@torch.no_grad()
def speculative_sampling_google_local_atten(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False,
                         start_attention_limit = 0, end_attention_limit = 0) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheLocalStreamingModel(approx_model, temperature, top_k, top_p,start_attention_limit,end_attention_limit)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        tot_num += 1#### 

        accepted_count_single_time = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
        
        accepted_num.extend([accepted_count_single_time])
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]
                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                break
                

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
    else:
        return prefix[:,seq_len:]    
    
    
@torch.no_grad()
def speculative_sampling_google_local_input(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, 
                         eos_token_id : int = None, output_count : bool = False,
                         start_attention_limit = 0, end_attention_limit = 0) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    add_bos_token = True
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        
        
        # Check if the prefix length is greater than the end_attention_limit
        if prefix_len-1 > end_attention_limit:
            # Calculate the start index for slicing
            start_index = prefix_len - end_attention_limit
            if add_bos_token :
            # Slice the prefix tensor to keep only the last end_attention_limit tensors
                truncated_prefix = torch.cat([prefix[:, :1], prefix[:, start_index:]], dim=1)
                approx_model_prefix_len = end_attention_limit + 1
            else:
                truncated_prefix = prefix[:, start_index:]
                approx_model_prefix_len = end_attention_limit
            
            approx_model_cache._past_key_values = None
            approx_model_cache._prob_history = None
        else:
            truncated_prefix = prefix
            approx_model_prefix_len = prefix_len
            
        
        
        x = approx_model_cache.generate(truncated_prefix, gamma)
        if prefix_len-1 > end_attention_limit:
            if add_bos_token :      
                x = torch.cat([prefix[:, :start_index], x[:, 1:]], dim=1)
            else:
                x = torch.cat([prefix[:, :start_index], x[:, :]], dim=1)
        
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        approx_model_n = approx_model_prefix_len + gamma - 1
        tot_num += 1#### 

        accepted_count_single_time = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, approx_model_prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                approx_model_n = approx_model_prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
        
        accepted_num.extend([accepted_count_single_time])
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(approx_model_n+1)
        
        
        assert approx_model_cache._prob_history.shape[-2] <= approx_model_n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {approx_model_n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, approx_model_n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]
                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                break
                

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num
        return SpModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                )        
    else:
        return prefix[:,seq_len:]        
    
    
    
    


@torch.no_grad()
def speculative_sampling_google_dynamic_gamma(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, eos_token_id : int = None, output_count : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    

    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    all_pass = False
    
    kl_div_out = []
    top_p = 0 
    top_k = 0
    kl_div_lambda = 0.5
    previous_gamma_num = 3
    epsilon = 1e-20
    input_gamma = gamma
    use_dynamic_gamma = False
    approx_target_kl = True
    gamma_sequence = []
    accepted_entropy = []
    reject_entropy = []
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        
        
        # print("=========================================================================")
        # if approx_model_cache._prob_history is not None :
        #     print(f"prefix_len:{prefix_len} ; before inference approx, the approx size :{approx_model_cache._prob_history.shape[1]}")
        # else:
        #     print(f"prefix_len:{prefix_len} ; before inference approx, the approx size :{0}")
        
        x = approx_model_cache.generate(prefix, gamma)
        # print(f"prefix_len:{prefix_len} ; after inference approx, the approx size :{approx_model_cache._prob_history.shape[1]}")
        
        
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        tot_num += 1

        accepted_count_single_time = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            
            
            

            # probabilities = approx_model_cache._prob_history[:, prefix_len + i - 1]
            # distribution = torch.distributions.Categorical(probs=probabilities)
            # entropy = distribution.entropy().item()
            
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                # reject_entropy.append(entropy)
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
            # accepted_entropy.append(entropy)
        
        accepted_num.extend([accepted_count_single_time])
        gamma_sequence.append(gamma)  
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"prefix_len:{prefix_len} ; target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
                    
            #######
            if approx_target_kl : 
                epsilon = 1e-10
                
                lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
                kl_div_out.append(F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :]+ epsilon, reduction='batchmean').item())
                kl_div_out_tensor = torch.tensor(kl_div_out)
                
                average_kl_div = torch.mean(kl_div_out_tensor).item()
                if use_dynamic_gamma :
                    if average_kl_div*kl_div_lambda < kl_div_out[-1] :
                        # gamma = 1
                        # gamma = max(1,int(sum(accepted_num[-previous_gamma_num:]) / len(accepted_num[-previous_gamma_num:]) + 0.5))
                        gamma = max(1,min(input_gamma,sum(accepted_num[-previous_gamma_num:])))  # method 3
                    else:
                        gamma = input_gamma
                # print(f"prefix_len:{prefix_len} ; partial pass n:{n}, approx size :{approx_model_cache._prob_history.shape[1]}")
                        
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
            
            # if approx_target_kl : 
            #     all_pass = True #########
                
            if approx_target_kl :
                lg_approx = torch.log(approx_model_cache._prob_history[:, n-1, :]+epsilon)
                ## use last time n
                kl_div_out.append(F.kl_div(lg_approx, target_model_cache._prob_history[:, n-1, :]+ epsilon, reduction='batchmean').item())           
                    # print(f"all pass n:{n}, approx size :{approx_model_cache._prob_history.shape[1]}")
                    
            if use_dynamic_gamma :
                if average_kl_div*kl_div_lambda < kl_div_out[-1] :
                    # gamma = 1 # method 1
                    # gamma = max(1,int(sum(accepted_num[-previous_gamma_num:]) / len(accepted_num[-previous_gamma_num:]) + 0.5)) # method 2
                    gamma = max(1,min(input_gamma,sum(accepted_num[-previous_gamma_num:])))  # method 3
                else:
                    gamma = input_gamma    
        
                             
        
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                                
                break



    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num , kl_div_out
        
        return SpDyGammaModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                kl_div_out = kl_div_out,
                gamma_sequence = gamma_sequence,
                accepted_entropy = accepted_entropy,
                reject_entropy = reject_entropy,
                )
    else:
        return prefix[:,seq_len:]





@torch.no_grad()
def speculative_sampling_google_dynamic_gamma_entropy(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4, golden_accepted_sequence : list = None,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, eos_token_id : int = None, output_count : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    # print(approx_model.device)
    # print(target_model.device)
    # breakpoint()
    
    device = target_model.device
    

    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    kl_div_out = []
    entropy_th = 4 
    ##recode entropy only
    entropy_th = 10e10     
    
    use_dynamic_gamma = False
    gamma_sequence = []
    accepted_entropy = []
    reject_entropy = []
    golden_accepted_sequence_temp = golden_accepted_sequence.copy() if golden_accepted_sequence is not None else None
    input_gamma = gamma    
    

    approx_model_cache = KVCacheModelEntropyGamma(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]    
        if golden_accepted_sequence_temp is not None:
            assert gamma >= 0, print("gamma < 0")
            if len(golden_accepted_sequence_temp) == 0 :
                gamma = input_gamma
            else:
                gamma = golden_accepted_sequence_temp.pop(0)
            if gamma < input_gamma :
                gamma = gamma + 1

        
        if gamma > 0 :
            x, gamma_out, entropys = approx_model_cache.generate(prefix, gamma, entropy_th)
            _ = target_model_cache.generate(x, 1)
        else:
            x = prefix
            gamma_out = gamma
            _ = target_model_cache.generate(x, 1)               
        
        
        n = prefix_len + gamma_out - 1
        tot_num += 1

        accepted_count_single_time = 0
        for i in range(gamma_out):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                reject_entropy.append(entropys[i])  ##recde only
                if use_dynamic_gamma :
                    reject_entropy.append(entropys[i])
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
            accepted_entropy.append(entropys[i]) ##recode only
            if use_dynamic_gamma :
                accepted_entropy.append(entropys[i])
        
        accepted_num.extend([accepted_count_single_time])
        gamma_sequence.append(gamma_out)  
        if use_dynamic_gamma :
            if len(accepted_entropy) > 0 and len(reject_entropy) > 0 :
                entropy_th = (sum(reject_entropy)/len(reject_entropy) + sum(accepted_entropy)/len(accepted_entropy))/2
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        if approx_model_cache._prob_history is not None:
            approx_model_cache.rollback(n+1)
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        if n < prefix_len + gamma_out - 1:
            # reject someone, sample from the pos n

            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"prefix_len:{prefix_len} ; target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)

        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
            
        
        prefix = torch.cat((prefix, t), dim=1)
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                                
                break



    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num , kl_div_out
        
        return SpDyGammaModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                kl_div_out = kl_div_out,
                gamma_sequence = gamma_sequence,
                accepted_entropy = accepted_entropy,
                reject_entropy = reject_entropy,
                )
    else:
        return prefix[:,seq_len:]



def speculative_sampling_google_dynamic_gamma_entropy_grad(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4, golden_accepted_sequence : list = None, 
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, eos_token_id : int = None, output_count : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    def detach_tensors(nested_tensor_list):
        if isinstance(nested_tensor_list, list):
            return [detach_tensors(t) for t in nested_tensor_list]
        elif isinstance(nested_tensor_list, tuple):
            return tuple(detach_tensors(t) for t in nested_tensor_list)
        elif isinstance(nested_tensor_list, torch.Tensor):
            return nested_tensor_list.detach()
        else:
            return nested_tensor_list


    
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    

    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    tot_num = 0 ####
    accepted_num = []
    kl_div_out = []
    entropy_th = 10e10 ######################################
    epsilon = 0
    
    use_dynamic_gamma = False
    gamma_sequence = []
    accepted_entropy = []
    reject_entropy = []
    golden_accepted_sequence_temp = golden_accepted_sequence.copy() if golden_accepted_sequence is not None else None
    input_gamma = gamma    
    step = 0
    approx_model.train()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(
        approx_model.parameters(),
        lr=1e-7,
    )        
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    

    approx_model_cache = KVCacheModelEntropyGammaGard(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
        
    while prefix.shape[1] < T:
        step += 1
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        
            
        if golden_accepted_sequence_temp is not None:
            assert gamma >= 0, print("gamma < 0")
            if len(golden_accepted_sequence_temp) == 0 :
                gamma = input_gamma
            else:
                gamma = golden_accepted_sequence_temp.pop(0)
            if gamma < input_gamma :
                gamma = gamma + 1

        
        if gamma > 0 :
            with torch.cuda.amp.autocast():
                x, gamma_out, entropys = approx_model_cache.generate(prefix, gamma, entropy_th)
            _ = target_model_cache.generate(x, 1)
        else:
            x = prefix
            gamma_out = gamma
            _ = target_model_cache.generate(x, 1)               
            
        n = prefix_len + gamma_out - 1
        tot_num += 1

        accepted_count_single_time = 0
        for i in range(gamma_out):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                # reject_entropy.append(entropy)
                if use_dynamic_gamma :
                    reject_entropy.append(entropys[i])
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            accepted_count_single_time += 1
            # accepted_entropy.append(entropy)
            if use_dynamic_gamma :
                accepted_entropy.append(entropys[i])
        
        accepted_num.extend([accepted_count_single_time])
        gamma_sequence.append(gamma_out)  
        if use_dynamic_gamma :
            if len(accepted_entropy) > 0 and len(reject_entropy) > 0 :
                entropy_th = (sum(reject_entropy)/len(reject_entropy) + sum(accepted_entropy)/len(accepted_entropy))/2
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        if approx_model_cache._prob_history is not None:
            approx_model_cache.rollback(n+1)
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        if n < prefix_len + gamma_out - 1:
            # reject someone, sample from the pos n

            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"prefix_len:{prefix_len} ; target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
            


            # ## all position and every step, wo deatch, lr_schedule
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='mean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     approx_model_cache._past_key_values = None  
            #     approx_model_cache._prob_history = None   

            # ## all position and every step, deatch, lr_schedule
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='mean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach() 

            # ## all position and every step, deatch
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='mean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach() 
                
            # ## n position and every step, deatch
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='batchmean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach() 
            
            
            # ##==== n position and each s step get the loss + update , deatch=====
            # s=20
            # if (resample_count) % s == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='batchmean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     # Detach past key values to clear grad_fn
            # approx_model_cache._past_key_values = [
            #     [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            # ]
            # approx_model_cache._prob_history = approx_model_cache._prob_history.detach()            
                
            # ##==== n position and each loss + each s step update , deatch========
            # s=10
            # with torch.cuda.amp.autocast():
            #     lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #     loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='batchmean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss/s).backward() 
            # if (resample_count) % s == 0 :                
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     # Detach past key values to clear grad_fn
            # approx_model_cache._past_key_values = [
            #     [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            # ]
            # approx_model_cache._prob_history = approx_model_cache._prob_history.detach()                                    



            ##==== all position and each s step get the loss + update , deatch=====
            s=20
            if (resample_count) % s == 0 :
                with torch.cuda.amp.autocast():
                    lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
                    loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='batchmean')              
                kl_div_out.append(loss.item())
                scaler.scale(loss).backward()     
                # Unscaling and checking for NaNs/INFs
                scaler.unscale_(optimizer)
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Detach past key values to clear grad_fn
            approx_model_cache._past_key_values = [
                [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            ]
            approx_model_cache._prob_history = approx_model_cache._prob_history.detach()  





            # ## all position and every step
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='mean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach()
                
                
            # ## single position (n) and every step
            # if (step + 1) % 1 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='batchmean')             
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach()
                                


            # ## all position (n) and s step
            # s=20
            # if (step + 1) % 20 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='mean')             
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     # Detach past key values to clear grad_fn
            #     approx_model_cache._past_key_values = [
            #         [tensor.detach() for tensor in inner_list] for inner_list in approx_model_cache._past_key_values
            #     ]
            #     approx_model_cache._prob_history = approx_model_cache._prob_history.detach()



            # # all position and each 10 step
            # if (step + 1) % 10 == 0 :
            #     with torch.cuda.amp.autocast():
            #         lg_approx = torch.log(approx_model_cache._prob_history+epsilon)
            #         loss = F.kl_div(lg_approx, target_model_cache._prob_history.detach()+ epsilon, reduction='mean')              
            #     kl_div_out.append(loss.item())
            #     scaler.scale(loss).backward()     
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     # lr_scheduler.step()
            #     approx_model_cache._past_key_values = None  
            #     approx_model_cache._prob_history = None   












            
            
            # with torch.cuda.amp.autocast():
            #     lg_approx = torch.log(approx_model_cache._prob_history[:, n, :]+epsilon)
            #     loss = F.kl_div(lg_approx, target_model_cache._prob_history[:, n, :].detach()+ epsilon, reduction='batchmean')              
            # # print(f"kl_div:{loss}, gamma:{gamma_out}, accetp time:{accepted_count_single_time}")   
            # # scaler.scale(loss/20).backward()     
            #     scaler.scale(loss/20).backward(retain_graph=True)        
            # if (step + 1) % 20 == 0 :
            #     # Unscaling and checking for NaNs/INFs
            #     scaler.unscale_(optimizer)
            #     # Apply gradient clipping to prevent exploding gradients
            #     torch.nn.utils.clip_grad_norm_(approx_model.parameters(), max_norm=1.0)                
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()
            #     approx_model_cache._past_key_values = None  
            #     approx_model_cache._prob_history = None              
  
                
            
                    
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
            
        
                            
        
        prefix = torch.cat((prefix, t), dim=1).detach()
        if eos_token_id is not None :
            if eos_token_id in prefix[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(prefix[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    prefix = prefix[:,:first_eos_index]             
                                
                break



    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print("============================={}===================".format(tot_num))####
    if output_count :
        # return prefix[:,seq_len:],accepted_count,target_sample_count,resample_count,accepted_num , kl_div_out
        
        return SpDyGammaModelOut(
                output_tokens = prefix[:,seq_len:],
                accepted_count = accepted_count,
                target_sample_count = target_sample_count,
                resample_count = resample_count,
                accepted_num = accepted_num,
                kl_div_out = kl_div_out,
                gamma_sequence = gamma_sequence,
                accepted_entropy = accepted_entropy,
                reject_entropy = reject_entropy,
                )
    else:
        return prefix[:,seq_len:]



