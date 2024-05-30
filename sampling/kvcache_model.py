import torch
from typing import Optional

from sampling.utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
import time

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]



class KVCacheModelStreaming():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._streaming_num = streaming_num

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                
                if self._streaming_num + 1  > end_pos :
                    
                    k = k[:, :, :end_pos, :]
                    v = v[:, :, :end_pos, :]
                else:
                    k = torch.cat((k[:, :, :1, :], k[:, :, (end_pos-self._streaming_num):end_pos, :]), dim=2)
                    v = torch.cat((v[:, :, :1, :], v[:, :, (end_pos-self._streaming_num):end_pos, :]), dim=2)
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
        
        
        



# class KVCacheModelStreamingOneModel():
#     def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
#         self._model = model
#         self._past_key_values = None
#         self._prob_history = None

#         self._temperature = temperature
#         self._top_k = top_k
#         self._top_p = top_p
#         self._streaming_num = streaming_num

#     def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
#         if self._past_key_values is None:
#             assert self._prob_history is None, f"{self._prob_history.shape}"
#             # the first forward (prefill) returns the prompt's logits
#             outputs = self._model(input_ids)
#             self._prob_history = outputs.logits
#             for i in range(self._prob_history.shape[-2]):   
#                 self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
#             self._past_key_values = outputs.past_key_values
#             last_q = self._prob_history[:, -1, :]
#         else:
#             # return the last token's logits
#             cached_len = 0
#             for kv in self._past_key_values:
#                 k, v = kv
#                 cached_len = k.shape[2]
                
#             last_input_id = input_ids[:, cached_len:]
#             if last_input_id.dim() == 1:
#                 last_input_id = torch.unsqueeze(last_input_id, 0)
            
#             if use_debug:
#                 print(f"last_input_id shape {last_input_id.shape}")
#                 _debug_show_kvcache(self._past_key_values)
            
#             outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
#             not_cached_q = outputs.logits
#             if not_cached_q.dim() == 2:
#                 not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
#             for i in range(not_cached_q.shape[-2]):   
#                 not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
#             self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
#             last_q = not_cached_q[:, -1, :]
#             self._past_key_values = outputs.past_key_values
        
#         return last_q


#     def _generate_with_kvcache(self, prefix : torch.Tensor, 
#                                     gamma : int, 
#                                     use_debug = False) -> torch.Tensor:
#         """ forward the model gamma times

#         Args:
#             prefix (torch.Tensor): the prefix
#             gamma (int): how many times approx guesses

#         Returns:
#             Torch.Tensor: prefix+generated tokens
#         """
#         x = prefix

#         for _ in range(gamma):
#             q = self._forward_with_kvcache(x, use_debug)
#             next_tok = sample(q)
#             x = torch.cat((x, next_tok), dim=1)
#         return x

#     @torch.no_grad()
#     def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
#         output = self._generate_with_kvcache(input, gamma)
#         return output
    
#     @torch.no_grad()
#     def rollback(self, end_pos : int):
#         past_key_values_trimmed = []
#         assert self._past_key_values
#         for kv in self._past_key_values:
#             k, v = kv
#             # NOTE() the indexing is specific for bloom. This won't work for other models
#             # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
#             # Bloom is special one
#             if isinstance(self._model, BloomForCausalLM):
#                 # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
#                 k = k[:, :, :end_pos]
#                 v = v[:, :end_pos, :]
#                 kv_trimmed = (k, v)
#                 past_key_values_trimmed.append(kv_trimmed)
#             else:
#                 # k, v (batch, head, seq, hidden_dim)
                
#                 if self._streaming_num + 1  > end_pos :
                    
#                     k = k[:, :, :end_pos, :]
#                     v = v[:, :, :end_pos, :]
#                 else:
#                     k = torch.cat((k[:, :, :1, :], k[:, :, (end_pos-self._streaming_num):end_pos, :]), dim=2)
#                     v = torch.cat((v[:, :, :1, :], v[:, :, (end_pos-self._streaming_num):end_pos, :]), dim=2)
#                 kv_trimmed = (k, v)
#                 past_key_values_trimmed.append(kv_trimmed)
        
#         self._past_key_values = past_key_values_trimmed
#         self._prob_history = self._prob_history[:, :end_pos, :]        
        
        




















class KVCacheModelStreamingOneModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._past_key_values

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p


    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits

            outputs = self._model(input_ids)
            
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
                
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q, self._prob_history
    

    def _forward_with_streaming_kvcache(self, input_ids : torch.Tensor, streaming_past_key_values: torch.Tensor, 
                                        streaming_prob_history: torch.Tensor , streaming_kv_start_pos:int, use_debug = True) -> torch.Tensor:
        
        assert streaming_prob_history is not None, f"{streaming_prob_history.shape}"
        # return the last token's logits
        cached_len = 0
        for kv in streaming_past_key_values:
            k, v = kv
            break
        cached_len = k.shape[2]
        last_input_id = input_ids[:, streaming_kv_start_pos + cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)
        
        if use_debug:
            print(f"last_input_id shape {last_input_id.shape}")
            _debug_show_kvcache(streaming_prob_history)
                
        # Generate position tensor starting from position_start_value
        position_ids = torch.arange(streaming_kv_start_pos + cached_len, input_ids.shape[1])

        # Move position tensor to the same device as input_ids
        position_ids = position_ids.unsqueeze(0).to(input_ids.device)   

        outputs = self._model(last_input_id, past_key_values=streaming_past_key_values, use_cache=True)
        
        
        # outputs = self._model(last_input_id, past_key_values=streaming_past_key_values,position_ids=position_ids, use_cache=True)
        streaming_past_key_values = outputs.past_key_values
        
        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
        for i in range(not_cached_q.shape[-2]):   
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
        streaming_prob_history = torch.cat([streaming_prob_history, not_cached_q], dim=1)
        
        last_q = not_cached_q[:, -1, :]

        
        return last_q, streaming_past_key_values, streaming_prob_history

    def process_streaming(self, x, gamma, streaming_num, use_debug):
        """
        Process streaming with given inputs.

        Parameters:
        - x: The input tensor.
        - gamma: The number of iterations to generate next tokens.
        - use_debug: Boolean flag to use debug mode or not.

        Returns:
        - x: The updated input tensor after processing.
        - streaming_prob_history: The updated streaming probability history.
        """
        
        
        streaming_past_key_values, streaming_prob_history, streaming_kv_start_pos = self.streaming_kv_prob(
            self._past_key_values, self._prob_history, streaming_num)

        # Generate new tokens for 'gamma' iterations
        for _ in range(gamma):
            q, streaming_past_key_values, streaming_prob_history = self._forward_with_streaming_kvcache(
                x, 
                streaming_past_key_values, 
                streaming_prob_history,
                streaming_kv_start_pos, 
                use_debug
            )
            next_tok = sample(q)  # Assuming 'sample' is a function that samples based on 'q'
            x = torch.cat((x, next_tok), dim=1)  # Concatenate the new token to 'x'
        
        # Return the updated 'x' and 'streaming_prob_history'
        return x, streaming_prob_history



    def _generate_with_streaming_kvcache(self, prefix: torch.Tensor, gamma: int, streaming_num:int, use_debug=False) -> torch.Tensor:
        """Generate tokens with streaming key-value cache.

        This method generates `gamma` additional tokens based on the given `prefix`,
        using a streaming key-value cache if applicable.

        Args:
            prefix (torch.Tensor): Initial tensor of tokens.
            gamma (int): Number of tokens to generate.
            use_debug (bool, optional): Flag for debugging. Defaults to False.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Tuple containing the tensor of generated tokens
                                                and the probability history.
        """
        x = prefix

        # Initialize an empty probability history

        # Generate tokens without streaming KV cache
        x, streaming_prob_history = self.process_streaming(x, gamma, streaming_num, use_debug)

        return x, streaming_prob_history 




    
    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        q, prob_history  = self._forward_with_kvcache(x, use_debug)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)
        return x,prob_history    

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int,  approx_model_action: bool,streaming_num:int) -> torch.Tensor:
        if approx_model_action:
            output, prob_history  = self._generate_with_streaming_kvcache(input, gamma, streaming_num)
        else:
            output, prob_history = self._generate_with_kvcache(input, gamma)
        return output, prob_history
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]  
        
    @torch.no_grad()
    def streaming_kv_prob(self, past_key_values,prob_history,streaming_num):
        past_key_values_trimmed = []
        for kv in past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = torch.cat((k[:, :, :1], k[:, :, -streaming_num+1:]), dim=2)
                v = torch.cat((v[:, :, :1], v[:, :, -streaming_num+1:]), dim=2)
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                if prob_history.shape[1] == streaming_num :
                    past_key_values_trimmed = past_key_values
                    prob_history = prob_history
                    streaming_kv_start_pos = 0
                    break
                else:
                    # k, v (batch, head, seq, hidden_dim)
                    k = torch.cat((k[:, :, :1, :], k[:, :, -streaming_num+1:, :]), dim=2)
                    v = torch.cat((v[:, :, :1, :], v[:, :, -streaming_num+1:, :]), dim=2)   
                    kv_trimmed = (k, v)
                    past_key_values_trimmed.append(kv_trimmed)
        streaming_kv_start_pos = prob_history.shape[1] - streaming_num

        prob_history= torch.cat((prob_history[:, :1, :], prob_history[:, -streaming_num+1:, :]), dim=1)
        return past_key_values_trimmed, prob_history, streaming_kv_start_pos      






















# class KVCacheModelStreamingOneModel():
#     def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
#         self._model = model
#         self._past_key_values = None
#         self._prob_history = None
#         self._past_key_values

#         self._temperature = temperature
#         self._top_k = top_k
#         self._top_p = top_p


#     def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
#         if self._past_key_values is None:
#             assert self._prob_history is None, f"{self._prob_history.shape}"
#             # the first forward (prefill) returns the prompt's logits
#             outputs = self._model(input_ids)
#             self._prob_history = outputs.logits
#             for i in range(self._prob_history.shape[-2]):   
#                 self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
#             self._past_key_values = outputs.past_key_values
#             last_q = self._prob_history[:, -1, :]
#         else:
#             # return the last token's logits
#             cached_len = 0
#             for kv in self._past_key_values:
#                 k, v = kv
#                 cached_len = k.shape[2]
#             last_input_id = input_ids[:, cached_len:]
#             if last_input_id.dim() == 1:
#                 last_input_id = torch.unsqueeze(last_input_id, 0)
            
#             if use_debug:
#                 print(f"last_input_id shape {last_input_id.shape}")
#                 _debug_show_kvcache(self._past_key_values)
            
#             outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
#             not_cached_q = outputs.logits
#             if not_cached_q.dim() == 2:
#                 not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
#             for i in range(not_cached_q.shape[-2]):   
#                 not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
#             self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
#             last_q = not_cached_q[:, -1, :]
#             self._past_key_values = outputs.past_key_values
        
#         return last_q, self._prob_history
    

#     def _forward_with_streaming_kvcache(self, input_ids : torch.Tensor, streaming_past_key_values: torch.Tensor, 
#                                         streaming_prob_history: torch.Tensor , streaming_kv_start_pos:int, use_debug = True) -> torch.Tensor:
        
#         assert streaming_prob_history is not None, f"{streaming_prob_history.shape}"
#         # return the last token's logits
#         cached_len = 0
#         for kv in streaming_past_key_values:
#             k, v = kv
#             break
#         cached_len = k.shape[2]
#         last_input_id = input_ids[:, streaming_kv_start_pos + cached_len:]
#         if last_input_id.dim() == 1:
#             last_input_id = torch.unsqueeze(last_input_id, 0)
        
#         if use_debug:
#             print(f"last_input_id shape {last_input_id.shape}")
#             _debug_show_kvcache(streaming_prob_history)
                
#         # Generate position tensor starting from position_start_value
#         position_ids = torch.arange(streaming_kv_start_pos + cached_len, input_ids.shape[1])

#         # Move position tensor to the same device as input_ids
#         position_ids = position_ids.unsqueeze(0).to(input_ids.device)   
             
#         outputs = self._model(last_input_id, past_key_values=streaming_past_key_values, use_cache=True)
#         # outputs = self._model(last_input_id, past_key_values=streaming_past_key_values,position_ids=position_ids, use_cache=True)
#         streaming_past_key_values = outputs.past_key_values
        
#         not_cached_q = outputs.logits
#         if not_cached_q.dim() == 2:
#             not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
#         for i in range(not_cached_q.shape[-2]):   
#             not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
#         streaming_prob_history = torch.cat([streaming_prob_history, not_cached_q], dim=1)
        
#         last_q = not_cached_q[:, -1, :]

        
#         return last_q, streaming_past_key_values, streaming_prob_history

#     def process_streaming(self, x, gamma, streaming_num, use_debug):
#         """
#         Process streaming with given inputs.

#         Parameters:
#         - x: The input tensor.
#         - gamma: The number of iterations to generate next tokens.
#         - use_debug: Boolean flag to use debug mode or not.

#         Returns:
#         - x: The updated input tensor after processing.
#         - streaming_prob_history: The updated streaming probability history.
#         """
#         streaming_past_key_values, streaming_prob_history, streaming_kv_start_pos = self.streaming_kv_prob(
#             self._past_key_values, self._prob_history, streaming_num)

#         # Generate new tokens for 'gamma' iterations
#         for _ in range(gamma):
#             q, streaming_past_key_values, streaming_prob_history = self._forward_with_streaming_kvcache(
#                 x, 
#                 streaming_past_key_values, 
#                 streaming_prob_history,
#                 streaming_kv_start_pos, 
#                 use_debug
#             )
#             next_tok = sample(q)  # Assuming 'sample' is a function that samples based on 'q'
#             x = torch.cat((x, next_tok), dim=1)  # Concatenate the new token to 'x'
        
#         # Return the updated 'x' and 'streaming_prob_history'
#         return x, streaming_prob_history



#     def _generate_with_streaming_kvcache(self, prefix: torch.Tensor, gamma: int, streaming_num:int, use_debug=False) -> torch.Tensor:
#         """Generate tokens with streaming key-value cache.

#         This method generates `gamma` additional tokens based on the given `prefix`,
#         using a streaming key-value cache if applicable.

#         Args:
#             prefix (torch.Tensor): Initial tensor of tokens.
#             gamma (int): Number of tokens to generate.
#             use_debug (bool, optional): Flag for debugging. Defaults to False.

#         Returns:
#             tuple(torch.Tensor, torch.Tensor): Tuple containing the tensor of generated tokens
#                                                 and the probability history.
#         """
#         x = prefix

#         # Initialize an empty probability history
#         prob_history = None

#         # Generate tokens without streaming KV cache
#         for _ in range(gamma):
#             # Process with streaming KV cache if conditions are met
#             if self._past_key_values is not None and streaming_num <= self._prob_history.shape[1]:
#                 x, streaming_prob_history = self.process_streaming(x, gamma, streaming_num, use_debug)
#                 return x, streaming_prob_history 
#             else:           
#                 q, prob_history = self._forward_with_kvcache(x, use_debug)
#                 next_tok = sample(q)  # Ensure 'sample' is defined or imported
#                 x = torch.cat((x, next_tok), dim=1)

#         return x, prob_history



    
#     def _generate_with_kvcache(self, prefix : torch.Tensor, 
#                                     gamma : int, 
#                                     use_debug = False) -> torch.Tensor:
#         """ forward the model gamma times

#         Args:
#             prefix (torch.Tensor): the prefix
#             gamma (int): how many times approx guesses

#         Returns:
#             Torch.Tensor: prefix+generated tokens
#         """
#         x = prefix
#         q, prob_history  = self._forward_with_kvcache(x, use_debug)
#         next_tok = sample(q)
#         x = torch.cat((x, next_tok), dim=1)
#         return x,prob_history    

#     @torch.no_grad()
#     def generate(self, input : torch.Tensor, gamma : int,  approx_model_action: bool,streaming_num:int) -> torch.Tensor:
#         if approx_model_action and streaming_num > 0:
#             output, prob_history  = self._generate_with_streaming_kvcache(input, gamma, streaming_num)
#         else:
#             output, prob_history = self._generate_with_kvcache(input, gamma)
#         return output, prob_history
    
#     @torch.no_grad()
#     def rollback(self, end_pos : int):
#         past_key_values_trimmed = []
#         assert self._past_key_values
#         for kv in self._past_key_values:
#             k, v = kv
#             # NOTE() the indexing is specific for bloom. This won't work for other models
#             # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
#             # Bloom is special one
#             if isinstance(self._model, BloomForCausalLM):
#                 # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
#                 k = k[:, :, :end_pos]
#                 v = v[:, :end_pos, :]
#                 kv_trimmed = (k, v)
#                 past_key_values_trimmed.append(kv_trimmed)
#             else:
#                 # k, v (batch, head, seq, hidden_dim)
#                 k = k[:, :, :end_pos, :]
#                 v = v[:, :, :end_pos, :]
#                 kv_trimmed = (k, v)
#                 past_key_values_trimmed.append(kv_trimmed)
        
#         self._past_key_values = past_key_values_trimmed
#         self._prob_history = self._prob_history[:, :end_pos, :]  
        
#     @torch.no_grad()
#     def streaming_kv_prob(self, past_key_values,prob_history,streaming_num):
#         past_key_values_trimmed = []
#         for kv in past_key_values:
#             k, v = kv
#             # NOTE() the indexing is specific for bloom. This won't work for other models
#             # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
#             # Bloom is special one
#             if isinstance(self._model, BloomForCausalLM):
#                 # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
#                 k = torch.cat((k[:, :, :1], k[:, :, -streaming_num+1:]), dim=2)
#                 v = torch.cat((v[:, :, :1], v[:, :, -streaming_num+1:]), dim=2)
#                 kv_trimmed = (k, v)
#                 past_key_values_trimmed.append(kv_trimmed)
#             else:
#                 if prob_history.shape[1] == streaming_num :
#                     past_key_values_trimmed = past_key_values
#                     prob_history = prob_history
#                     streaming_kv_start_pos = 0
#                     break
#                 else:
#                     # k, v (batch, head, seq, hidden_dim)
#                     k = torch.cat((k[:, :, :1, :], k[:, :, -streaming_num+1:, :]), dim=2)
#                     v = torch.cat((v[:, :, :1, :], v[:, :, -streaming_num+1:, :]), dim=2)   
#                     kv_trimmed = (k, v)
#                     past_key_values_trimmed.append(kv_trimmed)
#         streaming_kv_start_pos = prob_history.shape[1] - streaming_num

#         prob_history= torch.cat((prob_history[:, :1, :], prob_history[:, -streaming_num+1:, :]), dim=1)
#         return past_key_values_trimmed, prob_history, streaming_kv_start_pos      








class StreamingOneModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._past_key_values

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p


    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q, self._prob_history
    

    def _forward_with_streaming_kvcache(self, input_ids : torch.Tensor,appro_past_key_values = None, appro_prob_history=None, use_debug = True) -> torch.Tensor:
        if appro_past_key_values is None:
            assert appro_prob_history is None, f"{appro_prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            appro_prob_history = outputs.logits
            for i in range(appro_prob_history.shape[-2]):   
                appro_prob_history[:, i, :] = norm_logits(appro_prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            appro_past_key_values = outputs.past_key_values
            last_q = appro_prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in appro_past_key_values:
                k, v = kv
                cached_len = k.shape[2]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(appro_past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=appro_past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            appro_prob_history = torch.cat([appro_prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            appro_past_key_values = outputs.past_key_values
        
        return last_q, appro_past_key_values, appro_prob_history
    


    
    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        q, prob_history  = self._forward_with_kvcache(x, use_debug)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)
        return x,prob_history    
    
    
    def _generate_with_streaming_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    streaming_num : int,
                                    use_debug = False,) -> torch.Tensor:

        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        appro_past_key_values, appro_prob_history = None,None
        x = torch.cat((prefix[:,0], prefix[:,-streaming_num:]), dim=1)
        for _ in range(gamma):
            q, appro_past_key_values, appro_prob_history  = self._forward_with_streaming_kvcache(x, appro_past_key_values, appro_prob_history,use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x,appro_prob_history          
    
    

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int,  approx_model_action: bool,streaming_num:int) -> torch.Tensor:
        if approx_model_action:
            output, prob_history  = self._generate_with_streaming(input, gamma, streaming_num)
        else:
            output, prob_history = self._generate_with_kvcache(input, gamma)
        return output, prob_history
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]  
        
    @torch.no_grad()
    def streaming_kv_prob(self, past_key_values,prob_history,streaming_num):
        past_key_values_trimmed = []
        for kv in past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = torch.cat((k[:, :, :1], k[:, :, -streaming_num+1:]), dim=2)
                v = torch.cat((v[:, :, :1], v[:, :, -streaming_num+1:]), dim=2)
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                if prob_history.shape[1] == streaming_num :
                    past_key_values_trimmed = past_key_values
                    prob_history = prob_history
                    streaming_kv_start_pos = 0
                    break
                else:
                    # k, v (batch, head, seq, hidden_dim)
                    k = torch.cat((k[:, :, :1, :], k[:, :, -streaming_num+1:, :]), dim=2)
                    v = torch.cat((v[:, :, :1, :], v[:, :, -streaming_num+1:, :]), dim=2)   
                    kv_trimmed = (k, v)
                    past_key_values_trimmed.append(kv_trimmed)
        streaming_kv_start_pos = prob_history.shape[1] - streaming_num

        prob_history= torch.cat((prob_history[:, :1, :], prob_history[:, -streaming_num+1:, :]), dim=1)
        return past_key_values_trimmed, prob_history, streaming_kv_start_pos      







class test_time_Model():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0 , streaming_num : int = 4) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._past_key_values

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p


    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits

            outputs = self._model(input_ids)
            
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
                
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q, self._prob_history

    
    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        q, prob_history  = self._forward_with_kvcache(x, use_debug)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)
        return x,prob_history    

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int,  approx_model_action: bool,streaming_num:int) -> torch.Tensor:
        
        num_tokens = 10

        
        num_runs=10
        for num_tokens in range (10, 4000, 500) :
            total_time = 0
            input = torch.randint(0, 10, (1, num_tokens)).to(input.device)
            for j in range(num_runs):
                self._past_key_values,self._prob_history = None,None
                start_time = time.time() 
                input_1, prob_history = self._generate_with_kvcache(input, gamma)
                end_time = time.time()
                total_time += (end_time - start_time)
            average_time = total_time / num_runs
            print("First Time with {} tokens: {:.6f} seconds".format(num_tokens, average_time)) 
            first_past_key_values, first_prob_history = self._past_key_values,self._prob_history
            total_time = 0
            for j in range(num_runs):
                self._past_key_values,self._prob_history = first_past_key_values, first_prob_history
                start_time = time.time()  # Capture start time
                output, prob_history = self._generate_with_kvcache(input_1, gamma)
                end_time = time.time()  # Capture end time
                total_time += (end_time - start_time)
            average_time = total_time / num_runs
            print("KV Time with {} tokens: {:.6f} seconds".format(num_tokens, average_time))        
        return output, prob_history
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]  




class KVCacheLocalStreamingModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0,
                 start_attention_limit: int = 0,end_attention_limit: int = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.start_attention_limit = start_attention_limit
        self.end_attention_limit = end_attention_limit
        
    def create_local_mask(self, input_ids, start_attention_limit, end_attention_limit):
        """
        Creates an attention mask for the input_ids where the mask is set to 1
        for positions within the specified start and end attention limits. If both limits
        are set to zero, all positions are attended.

        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, seq_length]
            start_attention_limit (int): The number of initial tokens to include in attention.
            end_attention_limit (int): The number of final tokens to include in attention.

        Returns:
            torch.Tensor: An attention mask of shape [batch_size, seq_length]
        """
        # Get the batch size and sequence length from input_ids
        batch_size, seq_length = input_ids.shape

        # Get the device from input_ids
        device = input_ids.device

        # Special case: If both limits are zero, attend to all positions
        if start_attention_limit == 0 and end_attention_limit == 0:
            return torch.ones((batch_size, seq_length), dtype=torch.int, device=device)

        # Create a range tensor that mirrors the sequence positions
        range_tensor = torch.arange(seq_length, device=device).expand(batch_size, -1)

        # Create the mask by checking if each position is within the front and back attention limits
        attention_mask = ((range_tensor < start_attention_limit) | 
                        (range_tensor >= seq_length - end_attention_limit)).int()

        # Ensure the start-of-sequence token always receives attention
        attention_mask[:, 0] = 1

        return attention_mask 

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            attention_mask = self.create_local_mask(input_ids,self.start_attention_limit,self.end_attention_limit)
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True,attention_mask=attention_mask)
            
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]

            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
        
        



class KVCacheModelEntropyGamma():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, entropy_th : float,
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        entropys=[]
        for i in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
            
            distribution = torch.distributions.Categorical(probs=q)
            entropy = distribution.entropy().item()   
            entropys.append(entropy)
            if entropy > entropy_th :
                break 
           
        return x , i+1 ,entropys

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int, entropy_th : float) -> torch.Tensor:
        output, gamma_out, entropys = self._generate_with_kvcache(input, gamma, entropy_th)
        return output, gamma_out ,entropys
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
        
        
        
        
        
class KVCacheModelEntropyGammaGard(KVCacheModelEntropyGamma):
    
    def generate(self, input : torch.Tensor, gamma : int, entropy_th : float) -> torch.Tensor:
        output, gamma_out, entropys = self._generate_with_kvcache(input, gamma, entropy_th)
        return output, gamma_out ,entropys
    
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]    