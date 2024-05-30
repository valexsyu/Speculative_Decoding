import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, max_len : int, random_seed : int = None,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + max_len
    if random_seed:
        torch.manual_seed(random_seed)
    past_key_values = None
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
    return x


@torch.no_grad()
def autoregressive_sampling_with_eos(x : torch.Tensor, model : torch.nn.Module, max_len : int, random_seed : int = None,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, eos_token_id : int = None):
    n = len(x)
    T = len(x) + max_len

    if random_seed:
        torch.manual_seed(random_seed)

    past_key_values = None
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
        
        if eos_token_id is not None :
            if eos_token_id in x[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(x[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    x = x[:,:first_eos_index]             
                
                break        
    return x