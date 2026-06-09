# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""PyTorch Pangu V2 MoE model."""

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, logging
from .configuration_pangu_v2_moe import PanguUltraMoEConfig


logger = logging.get_logger(__name__)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_interleave(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, interleaved=False):
    rotate_fn = rotate_interleave if interleaved else rotate_half
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_fn(q) * sin), (k * cos) + (rotate_fn(k) * sin)


class PanguV2MoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PanguV2MoERotaryEmbedding(nn.Module):
    def __init__(self, config: PanguUltraMoEConfig):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.base = config.rope_theta

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
        )
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class PanguV2MoEMLP(nn.Module):
    def __init__(self, config: PanguUltraMoEConfig, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PanguV2MoESparseMoeBlock(nn.Module):
    def __init__(self, config: PanguUltraMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        if config.router_enable_expert_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        else:
            self.register_parameter("e_score_correction_bias", None)

        self.experts = nn.ModuleList(
            [PanguV2MoEMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = PanguV2MoEMLP(config, intermediate_size=shared_intermediate)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(hidden_states.to(self.gate.weight.dtype), self.gate.weight).to(torch.float32)
        routing_weights_all = torch.sigmoid(router_logits)

        routing_scores = routing_weights_all
        if self.e_score_correction_bias is not None:
            routing_scores = routing_scores + self.e_score_correction_bias
        _, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)
        routing_weights = torch.gather(routing_weights_all, dim=-1, index=selected_experts)

        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor is not None:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


class PanguV2MoEMHC(nn.Module):
    def __init__(self, config: PanguUltraMoEConfig, pre_only=False):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        self.hc_eps = 1e-6
        self.pre_only = pre_only

        out_features = self.num_stream if pre_only else (self.num_stream + 2) * self.num_stream
        self.phi = nn.Linear(self.hidden_size * self.num_stream, out_features, bias=False)
        self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.float32))
        if pre_only:
            self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.float32))
        else:
            self.branch_alpha = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.branch_beta = nn.Parameter(torch.empty(self.num_stream * (self.num_stream + 2), dtype=torch.float32))

    def _normed_mix(self, hidden_states):
        flat = hidden_states.flatten(-2).to(torch.float32)
        variance = flat.pow(2).mean(-1, keepdim=True)
        flat = flat * torch.rsqrt(variance + self.norm_eps) * self.norm_gamma
        return self.phi(flat.to(self.phi.weight.dtype)).to(torch.float32)

    def mhc_pre(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(*hidden_states.shape[:-2], self.num_stream, self.hidden_size)
        mixes = self._normed_mix(hidden_states)
        if self.pre_only:
            h_pre = torch.sigmoid(mixes * self.branch_alpha_pre + self.branch_beta_pre) + self.hc_eps
            hidden_states = torch.sum(h_pre.unsqueeze(-1) * hidden_states.to(torch.float32), dim=-2)
            return hidden_states.to(input_dtype), None, None

        h_pre, h_post, h_res = mixes.split([self.num_stream, self.num_stream, self.num_stream**2], dim=-1)
        beta_pre, beta_post, beta_res = self.branch_beta.split(
            [self.num_stream, self.num_stream, self.num_stream**2], dim=-1
        )
        h_pre = torch.sigmoid(h_pre * self.branch_alpha[0] + beta_pre) + self.hc_eps
        h_post = 2 * torch.sigmoid(h_post * self.branch_alpha[1] + beta_post)
        h_res = h_res.view(*hidden_states.shape[:-2], self.num_stream, self.num_stream)
        h_res = h_res * self.branch_alpha[2] + beta_res.view(self.num_stream, self.num_stream)
        hidden_states = torch.sum(h_pre.unsqueeze(-1) * hidden_states.to(torch.float32), dim=-2)
        return hidden_states.to(input_dtype), h_post, h_res

    def mhc_sinkhorn(self, h_res):
        if h_res is None:
            return None
        h_res = h_res.softmax(-1) + self.hc_eps
        h_res = h_res / (h_res.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(max(self.mhc_recur_norm - 1, 0)):
            h_res = h_res / (h_res.sum(-1, keepdim=True) + self.hc_eps)
            h_res = h_res / (h_res.sum(-2, keepdim=True) + self.hc_eps)
        return h_res

    def mhc_post(self, hidden_states, h_post, residual, h_res):
        if self.pre_only:
            return residual
        residual = residual.view(*residual.shape[:-2], self.num_stream, self.hidden_size)
        hidden_states = (
            h_post.unsqueeze(-1) * hidden_states.unsqueeze(-2)
            + torch.sum(h_res.unsqueeze(-1) * residual.unsqueeze(-3), dim=-2)
        )
        return hidden_states.to(residual.dtype)


class PanguV2MoEAttention(nn.Module):
    def __init__(self, config: PanguUltraMoEConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.attention_dropout = config.attention_dropout
        self.param_sink_number = config.param_sink_number

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = PanguV2MoERMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = PanguV2MoERMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        if self.param_sink_number > 0:
            self.param_sink_compressed_kv = nn.Parameter(torch.empty(self.param_sink_number, self.kv_lora_rank))
            self.param_sink_k_pe = nn.Parameter(torch.empty(self.param_sink_number, self.qk_rope_head_dim))
        else:
            self.register_parameter("param_sink_compressed_kv", None)
            self.register_parameter("param_sink_k_pe", None)

    def _project_qkv(self, hidden_states, cos, sin):
        batch_size, sequence_length, _ = hidden_states.shape
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, sequence_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        k_nope, value_states = kv.split(
            [self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim], dim=-1
        )
        k_nope = k_nope.view(batch_size, sequence_length, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        k_pe = k_pe.view(batch_size, sequence_length, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        value_states = value_states.view(batch_size, sequence_length, self.num_heads, self.v_head_dim).transpose(1, 2)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, self.config.rope_interleaved)
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)
        return query_states, key_states, value_states

    def _sink_kv(self, batch_size, dtype, device):
        if self.param_sink_number <= 0:
            return None, None
        compressed = self.kv_a_layernorm(self.param_sink_compressed_kv.to(device=device, dtype=dtype))
        sink_kv = self.kv_b_proj(compressed)
        sink_k_nope, sink_v = sink_kv.split(
            [self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim], dim=-1
        )
        sink_k_nope = sink_k_nope.view(self.param_sink_number, self.num_heads, self.qk_nope_head_dim).permute(1, 0, 2)
        sink_k_pe = self.param_sink_k_pe.to(device=device, dtype=dtype).view(
            self.param_sink_number, 1, self.qk_rope_head_dim
        )
        sink_k_pe = sink_k_pe.expand(-1, self.num_heads, -1).permute(1, 0, 2)
        sink_key = torch.cat([sink_k_nope, sink_k_pe], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        sink_value = sink_v.view(self.param_sink_number, self.num_heads, self.v_head_dim).permute(1, 0, 2)
        sink_value = sink_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return sink_key, sink_value

    def _sliding_window(self):
        if self.layer_idx not in self.config.swa_layers:
            return None
        return self.config.sliding_window_list[self.config.swa_layers.index(self.layer_idx)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[Cache] = None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        batch_size, query_length, _ = hidden_states.shape
        cos, sin = position_embeddings
        query_states, key_states, value_states = self._project_qkv(hidden_states, cos, sin)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        is_prefill = past_seen_tokens == 0
        if is_prefill:
            sink_key, sink_value = self._sink_kv(batch_size, key_states.dtype, key_states.device)
            if sink_key is not None:
                key_states = torch.cat([sink_key, key_states], dim=2)
                value_states = torch.cat([sink_value, value_states], dim=2)

        if use_cache and past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.qk_head_dim)
        key_length = key_states.shape[-2]
        sink_length = self.param_sink_number if self.param_sink_number > 0 and key_length > query_length else 0
        token_key_length = key_length - sink_length

        key_positions = torch.arange(token_key_length, device=hidden_states.device)
        query_positions = cache_position if cache_position is not None else torch.arange(query_length, device=hidden_states.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        sliding_window = self._sliding_window()
        if sliding_window is not None:
            causal_mask = causal_mask & (key_positions[None, :] > query_positions[:, None] - sliding_window)
        if sink_length:
            sink_mask = torch.ones(query_length, sink_length, dtype=torch.bool, device=hidden_states.device)
            causal_mask = torch.cat([sink_mask, causal_mask], dim=-1)
        causal_mask = causal_mask.view(1, 1, query_length, key_length)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                padding_mask = attention_mask[:, None, None, :].bool()
                if sink_length:
                    sink_padding = torch.ones(batch_size, 1, 1, sink_length, dtype=torch.bool, device=attention_mask.device)
                    padding_mask = torch.cat([sink_padding, padding_mask], dim=-1)
                causal_mask = causal_mask & padding_mask[:, :, :, -key_length:]
            elif attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask[:, :, :, -key_length:]

        min_dtype = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~causal_mask, min_dtype)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, self.num_heads * self.v_head_dim
        )
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class PanguV2MoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PanguUltraMoEConfig, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = PanguV2MoEAttention(config, layer_idx=layer_idx)
        if config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = PanguV2MoESparseMoeBlock(config)
        else:
            self.mlp = PanguV2MoEMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = PanguV2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PanguV2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = PanguV2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = PanguV2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sandwich_norm = config.sandwich_norm
        self.use_mhc = config.mhc_num_stream > 1 and layer_idx < config.num_hidden_layers
        if self.use_mhc:
            self.attn_mhc_module = PanguV2MoEMHC(config, pre_only=False)
            self.mlp_mhc_module = PanguV2MoEMHC(config, pre_only=False)
        self.use_post_norm = self.use_mhc and layer_idx in config.block_post_layernorm_idx
        if self.use_post_norm:
            self.block_post_layernorm = PanguV2MoERMSNorm(
                config.hidden_size * config.mhc_num_stream, eps=config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        if self.use_mhc:
            hidden_states, h_post, h_res = self.attn_mhc_module.mhc_pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_mhc:
            h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
            hidden_states = self.attn_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        if self.use_mhc:
            hidden_states, h_post, h_res = self.mlp_mhc_module.mhc_pre(hidden_states)
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        router_logits = None
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        hidden_states = self.post_mlp_layernorm(hidden_states)
        if self.use_mhc:
            h_res = self.mlp_mhc_module.mhc_sinkhorn(h_res)
            hidden_states = self.mlp_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        else:
            hidden_states = residual + hidden_states

        if self.use_post_norm:
            original_shape = hidden_states.shape
            hidden_states = self.block_post_layernorm(hidden_states.flatten(-2)).view(original_shape)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


@auto_docstring
class PanguV2MoEPreTrainedModel(PreTrainedModel):
    config: PanguUltraMoEConfig
    config_class = PanguUltraMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanguV2MoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, PanguV2MoERMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, PanguV2MoEMHC):
            module.norm_gamma.data.normal_(mean=1.0, std=std)
            if module.pre_only:
                module.branch_alpha_pre.data.zero_()
                module.branch_beta_pre.data.zero_()
            else:
                module.branch_alpha.data.zero_()
                module.branch_beta.data.zero_()


@auto_docstring
class PanguV2MoEModel(PanguV2MoEPreTrainedModel):
    def __init__(self, config: PanguUltraMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PanguV2MoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PanguV2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PanguV2MoERotaryEmbedding(config)
        self.use_mhc = config.mhc_num_stream > 1
        if self.use_mhc:
            self.merge_mhc_module = PanguV2MoEMHC(config, pre_only=True)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if past_seen_tokens > self.config.param_sink_number:
                past_seen_tokens -= self.config.param_sink_number
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        if self.use_mhc:
            hidden_states = hidden_states.unsqueeze(-2).expand(
                *hidden_states.shape[:-1], self.config.mhc_num_stream, self.config.hidden_size
            )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            output_index = 1
            if output_attentions:
                all_self_attns += (layer_outputs[output_index],)
                output_index += 1
            if output_router_logits and layer_outputs[output_index] is not None:
                all_router_logits += (layer_outputs[output_index],)

        if self.use_mhc:
            hidden_states, _, _ = self.merge_mhc_module.mhc_pre(hidden_states)
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    routing_weights = torch.sigmoid(concatenated_gate_logits)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts


@auto_docstring
class PanguV2MoEForCausalLM(PanguV2MoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = PanguV2MoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


PanguUltraMoEForCausalLM = PanguV2MoEForCausalLM
PanguUltraMoEModel = PanguV2MoEModel


__all__ = [
    "PanguUltraMoEForCausalLM",
    "PanguUltraMoEModel",
    "PanguV2MoEForCausalLM",
    "PanguV2MoEModel",
    "PanguV2MoEPreTrainedModel",
]
