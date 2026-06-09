# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pangu V2 MoE model configuration."""

from ...configuration_utils import PretrainedConfig


class PanguUltraMoEConfig(PretrainedConfig):
    r"""
    Configuration class for Pangu V2 MoE checkpoints whose remote config class is commonly named
    `PanguUltraMoEConfig` and whose architecture is `PanguV2MoEForCausalLM`.
    """

    model_type = "pangu_v2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_a_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "colwise",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=5120,
        intermediate_size=16128,
        num_hidden_layers=50,
        num_attention_heads=64,
        hidden_act="silu",
        max_position_embeddings=524288,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        rope_theta=6400000,
        rope_scaling=None,
        rope_interleaved=False,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        first_k_dense_replace=3,
        moe_intermediate_size=1792,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        router_enable_expert_bias=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        dsa_layers=None,
        swa_layers=None,
        sliding_window_list=None,
        index_topk=0,
        index_n_heads=None,
        index_head_dim=None,
        param_sink_number=0,
        param_sink_with_value=False,
        sandwich_norm=True,
        use_mhc=False,
        mhc_use_gamma=True,
        mhc_recur_norm=20,
        mhc_num_stream=1,
        block_post_layernorm_idx=None,
        use_mome=False,
        router_sliding_window=None,
        router_sliding_windows=None,
        num_nextn_predict_layers=0,
        pad_token_id=None,
        bos_token_id=148899,
        eos_token_id=148902,
        architectures=None,
        **kwargs,
    ):
        if router_sliding_window is None and router_sliding_windows is not None:
            router_sliding_window = router_sliding_windows

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleaved = rope_interleaved

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.router_enable_expert_bias = router_enable_expert_bias
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.num_experts = n_routed_experts

        self.dsa_layers = [] if dsa_layers is None else dsa_layers
        self.swa_layers = [] if swa_layers is None else swa_layers
        self.sliding_window_list = [] if sliding_window_list is None else sliding_window_list
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.param_sink_number = param_sink_number
        self.param_sink_with_value = param_sink_with_value

        self.sandwich_norm = sandwich_norm
        self.use_mhc = use_mhc
        self.mhc_use_gamma = mhc_use_gamma
        self.mhc_recur_norm = mhc_recur_norm
        self.mhc_num_stream = mhc_num_stream
        self.block_post_layernorm_idx = [] if block_post_layernorm_idx is None else block_post_layernorm_idx
        self.use_mome = use_mome
        self.router_sliding_window = router_sliding_window
        self.router_sliding_windows = router_sliding_windows
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.architectures = ["PanguV2MoEForCausalLM"] if architectures is None else architectures

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


PanguV2MoEConfig = PanguUltraMoEConfig


__all__ = ["PanguUltraMoEConfig", "PanguV2MoEConfig"]
