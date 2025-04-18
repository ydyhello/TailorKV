import copy
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch
from packaging import version
import time
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    is_hqq_available,
    is_optimum_quanto_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer
import time
from dgl.utils import gather_pinned_tensor_rows

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

import math
from torch import nn
from concurrent.futures import ThreadPoolExecutor
# max_workers = os.cpu_count()
logger = logging.get_logger(__name__)


import gc
# MODEL_NAME_TO_QUANT_LATERS = {
#     "NousResearch/Meta-Llama-3.1-8B-Instruct":[0],
#     "togethercomputer/Llama-2-7B-32K-Instruct":[0,1],
#     "01-ai/Yi-9B-200K":[0,1],
#     "01-ai/Yi-6B-200K":[0,1],
#     "mistralai/Mistral-7B-Instruct-v0.3":[0],
#     "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4":[0],
#     "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4":[0],
# }
MODEL_NAME_TO_QUANT_LATERS = json.load(open("config/model2quantlayer.json", "r"))

class TailorKVCache(Cache):

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(
        self, 
        layers, 
        config: PretrainedConfig, 
        max_batch_size: int,
        max_cache_len: Optional[int],
        input_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        num_hidden_layers: Optional[int] = None,
        ) -> None:

        super().__init__()
        
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        # self.key_cache: List[torch.Tensor] = []
        # self.value_cache: List[torch.Tensor] = []
        self._gen_tokens = 0
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float16

        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.input_len = input_len
        self.budget = config.budget
        self.recent_size = config.recent_size
        self.start_size = config.start_size
        assert self.recent_size + self.start_size < input_len
        self.num_channel = config.num_channel


        if self.budget > input_len:
            self.budget = input_len
        self.cache_size = self.budget-self.recent_size-self.start_size

        cache_shape = (max_batch_size, config.num_key_value_heads, self.max_cache_len, head_dim)

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.quant_layer = MODEL_NAME_TO_QUANT_LATERS[config._name_or_path]

        for i in range(config.num_hidden_layers):
            device = self.device if (i in self.quant_layer) else torch.device("cpu")
            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, device)
            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)
        
        trans_cache_shape = (max_batch_size, config.num_key_value_heads, head_dim, self.max_cache_len)
        
        self.trans_key_cache: List[torch.Tensor] = []

        for i in range(config.num_hidden_layers):
            device = self.device if (i in self.quant_layer) else torch.device("cpu")
            trans_key_cache= self._create_trans_key_cache_tensors(trans_cache_shape, device)
            self.trans_key_cache.append(trans_key_cache)

        self.max_gen_len = config.max_gen_len

        self.partial_key_cache_shape = (max_batch_size, config.num_key_value_heads, self.num_channel, self.max_cache_len+self.max_gen_len)
        self.partial_key_cache: List[torch.Tensor] = []
        
        for i in range(2):
            key_cache = self._create_partial_key_cache_tensors(self.partial_key_cache_shape, self.device)
            self.partial_key_cache.append(key_cache)

        self._device_key_cache = torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            self.budget,
            head_dim,
            device=self.device,
            dtype=self.dtype
        ) 

        self._device_value_cache = torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            self.budget,
            head_dim,
            device=self.device,
            dtype=self.dtype
            )

        self.device_channel_top_indices = [
            torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            1,
            self.num_channel,
            device=self.device,
            dtype=torch.int64)
            for _ in range(config.num_hidden_layers)
        ]

        self.max_key_states = [
            torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            1,
            head_dim,
            device=self.device,
            dtype=self.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.static_key_cache = [
            torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            self.recent_size + self.start_size,
            head_dim,
            device=self.device,
            dtype=self.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.static_value_cache = [
            torch.zeros(
            max_batch_size,
            config.num_key_value_heads,
            self.recent_size + self.start_size,
            head_dim,
            device=self.device,
            dtype=self.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.gen_key_cache = [
            torch.zeros(
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_gen_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.gen_value_cache = [
            torch.zeros(
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_gen_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.label_index_prefix = torch.arange(0, max_batch_size * config.num_key_value_heads, device=self.device) * self.max_cache_len
        self.channel_index_prefix = torch.arange(0, max_batch_size * config.num_key_value_heads, device=self.device) * head_dim

        self.layers = layers

        self.event = torch.cuda.Event(enable_timing=False)
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.future = [None for _ in range(self.num_layers)]

        self.residual_length = config.residual_length
        self.group_size = config.group_size
        self.v_bits = config.v_bits
        self.k_bits = config.k_bits
        self.kvquant_unit: list[Tuple[torch.Tensor]] = [None] * config.num_hidden_layers
        torch.cuda.empty_cache()


    def _create_partial_key_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor]:
        is_cpu_device = device == torch.device("cpu")
        key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        torch._dynamo.mark_static_address(key_cache)
        return key_cache

    def _create_key_value_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_cpu_device = device == torch.device("cpu")
        if is_cpu_device:
            key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
            value_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)

            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            torch._dynamo.mark_static_address(key_cache)
            torch._dynamo.mark_static_address(value_cache)
        else:
            key_cache = None
            value_cache = None
        return key_cache, value_cache

    def _create_trans_key_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor]:
        is_cpu_device = device == torch.device("cpu")
        if is_cpu_device:
            key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            torch._dynamo.mark_static_address(key_cache)
        else:
            key_cache = None
        return key_cache

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def save_kv_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
        kv_len = key_states.shape[-2]
        if kv_len != 1 and layer_idx not in self.quant_layer:

            self.key_cache[layer_idx].copy_(key_states,non_blocking=True)
            self.value_cache[layer_idx].copy_(value_states,non_blocking=True)

            self.trans_key_cache[layer_idx][:,:,:,:self._seen_tokens].copy_(key_states.transpose(2, 3).contiguous(),non_blocking=True)

            self.static_key_cache[layer_idx][:,:,:self.start_size,:] = key_states[:,:,:self.start_size,:].clone()
            self.static_key_cache[layer_idx][:,:,-self.recent_size:,:] = key_states[:,:,-self.recent_size:,:].clone()
            self.static_value_cache[layer_idx][:,:,:self.start_size,:] = value_states[:,:,:self.start_size,:].clone()
            self.static_value_cache[layer_idx][:,:,-self.recent_size:,:] = value_states[:,:,-self.recent_size:,:].clone()
        elif kv_len == 1 and layer_idx not in self.quant_layer:
            self.gen_key_cache[layer_idx][:,:,self._gen_tokens-1:self._gen_tokens,:] = key_states
            self.gen_value_cache[layer_idx][:,:,self._gen_tokens-1:self._gen_tokens,:] = value_states
        else:
            attn_output = self.quantize(layer_idx, key_states, value_states, query_states)
            return attn_output
            
    
    def quantize(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
        if key_states.shape[-2] != 1:
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), 
                                                                                                                self.group_size,self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None

            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)

            self.kvquant_unit[layer_idx] = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn)

        else:
            key_states_quant_trans = self.kvquant_unit[layer_idx][0]
            key_states_full = self.kvquant_unit[layer_idx][1]
            key_scale_trans = self.kvquant_unit[layer_idx][2]
            key_mn_trans = self.kvquant_unit[layer_idx][3]
            value_states_quant = self.kvquant_unit[layer_idx][4]
            value_states_full = self.kvquant_unit[layer_idx][5]
            value_scale = self.kvquant_unit[layer_idx][6]
            value_mn = self.kvquant_unit[layer_idx][7]       
            quant_key_len = key_states_quant_trans.shape[-1] * 32 // self.k_bits

            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
            
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)
            
            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))
                
            attn_output = attn_output.transpose(1, 2).contiguous()

            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

            self.kvquant_unit[layer_idx] = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn)
            return attn_output

    def load_gpu(self, layer_idx: int, flatten_index):
        self._device_key_cache[:,:,:self.start_size+self.recent_size,:] = self.static_key_cache[layer_idx]
        self._device_value_cache[:,:,:self.start_size+self.recent_size,:] = self.static_value_cache[layer_idx]
        
        D = self.head_dim

        self._device_key_cache[:,:,self.start_size+self.recent_size:,:] = gather_pinned_tensor_rows(self.key_cache[layer_idx].view(-1, D), flatten_index).view(self.max_batch_size,self.num_key_value_heads,self.cache_size,D)
        self._device_value_cache[:,:,self.start_size+self.recent_size:,:] = gather_pinned_tensor_rows(self.value_cache[layer_idx].view(-1, D), flatten_index).view(self.max_batch_size,self.num_key_value_heads,self.cache_size,D)
        
        compress_k = torch.cat([self._device_key_cache, self.gen_key_cache[layer_idx][:,:,:self._gen_tokens,:]],dim=2)
        compress_v = torch.cat([self._device_value_cache, self.gen_value_cache[layer_idx][:,:,:self._gen_tokens,:]],dim=2)

        return compress_k,compress_v
    

    def prefecth_partial_k(self, next_layer_query_states, layer_idx: int):
        if (next_layer_query_states is not None) and ((layer_idx+1) not in self.quant_layer):
            channel_result = torch.mul(torch.abs(reshape_kv(next_layer_query_states,self.num_key_value_groups)), self.max_key_states[layer_idx+1])
            _, channel_top_indices = torch.topk(channel_result, self.num_channel, dim=-1, sorted=False) 
            self.device_channel_top_indices[layer_idx+1] = channel_top_indices 

            D = self.max_cache_len

            channel_flatten_index = self.channel_index_prefix[:,None] + channel_top_indices.view(self.max_batch_size * self.num_key_value_heads, self.num_channel)      
            self.partial_key_cache[(layer_idx+1) & 1][:,:,:,:self.input_len] = gather_pinned_tensor_rows(self.trans_key_cache[layer_idx+1].view(-1, D), channel_flatten_index.view(-1)).view(self.max_batch_size, self.num_key_value_heads, self.num_channel, D)


    def approximate_attn(self, query_states, key_states, layer_idx: int):
        channel_query_states = torch.gather(reshape_kv(query_states,self.num_key_value_groups), dim=-1, index=self.device_channel_top_indices[layer_idx].expand(-1, -1, 1, -1))
        channel_key_states = torch.gather(self.gen_key_cache[layer_idx][:,:,:self._gen_tokens,:],dim=-1,index=self.device_channel_top_indices[layer_idx].expand(-1, -1, self._gen_tokens, -1)).transpose(2, 3).contiguous()
        self.partial_key_cache[layer_idx & 1][:,:,:,self.input_len:self._seen_tokens] = channel_key_states
        partial_att = torch.matmul(channel_query_states,self.partial_key_cache[layer_idx & 1][:,:,:,:self._seen_tokens])
        _, topk_indices = torch.topk(partial_att[:,:,:,self.start_size:-self.recent_size-self._gen_tokens], k=self.cache_size, dim=-1, sorted=False)
        topk_indices = topk_indices + self.start_size
        return topk_indices
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        next_layer_query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            if key_states.shape[-2] == 1:
                self._gen_tokens += 1

        if key_states.shape[-2] != 1:
            self.save_kv_cache(layer_idx, key_states, value_states, query_states)
            self.max_key_states[layer_idx] = torch.max(torch.abs(key_states), dim=2, keepdim=True)[0]
            torch.cuda.empty_cache()
            
        else:
            if layer_idx not in self.quant_layer:
                self.save_kv_cache(layer_idx, key_states, value_states, query_states)
            else:
                self.future[(layer_idx+1) % self.num_layers] = self.executor.submit(self.prefecth_partial_k, next_layer_query_states, layer_idx)
                attn_output = self.save_kv_cache(layer_idx, key_states, value_states, query_states)
                return attn_output

            self.future[layer_idx].result()
            topk_indices = self.approximate_attn(query_states, key_states, layer_idx)
            
            flatten_index = self.label_index_prefix[:,None] + topk_indices.view(self.max_batch_size * self.num_key_value_heads, self.cache_size)
            compress_k,compress_v = self.load_gpu(layer_idx, flatten_index.view(-1))  
            self.future[(layer_idx+1) % self.num_layers] = self.executor.submit(self.prefecth_partial_k, next_layer_query_states, layer_idx)

            return compress_k, compress_v



    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def reshape_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.view(batch, num_attention_heads // n_rep, n_rep, slen, head_dim)
    return hidden_states.mean(dim=2)