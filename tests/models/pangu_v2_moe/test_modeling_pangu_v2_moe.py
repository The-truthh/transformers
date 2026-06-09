# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Testing suite for the PyTorch Pangu V2 MoE model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        PanguUltraMoEConfig,
        PanguV2MoEConfig,
        PanguV2MoEForCausalLM,
        PanguV2MoEModel,
    )


def get_pangu_v2_moe_config(**kwargs):
    config_kwargs = {
        "vocab_size": 99,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "max_position_embeddings": 64,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "param_sink_number": 4,
        "param_sink_with_value": True,
        "sandwich_norm": True,
        "use_mhc": True,
        "mhc_num_stream": 2,
        "mhc_recur_norm": 2,
        "mhc_use_gamma": True,
        "block_post_layernorm_idx": [0],
        "swa_layers": [1, 2],
        "sliding_window_list": [8, 8],
        "index_topk": 8,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "output_router_logits": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    config_kwargs.update(kwargs)
    return PanguUltraMoEConfig(**config_kwargs)


@require_torch
class PanguV2MoEModelTest(unittest.TestCase):
    def get_input_ids(self, batch_size=2, seq_length=5, vocab_size=99):
        return torch.arange(batch_size * seq_length, dtype=torch.long, device=torch_device).view(
            batch_size, seq_length
        ) % vocab_size

    def test_config_aliases_and_auto_classes(self):
        config = AutoConfig.for_model("pangu_v2_moe", vocab_size=99)
        self.assertIsInstance(config, PanguUltraMoEConfig)
        self.assertIs(PanguV2MoEConfig, PanguUltraMoEConfig)

        config = get_pangu_v2_moe_config()
        self.assertIsInstance(AutoModel.from_config(config), PanguV2MoEModel)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), PanguV2MoEForCausalLM)

    def test_router_sliding_windows_alias(self):
        config = get_pangu_v2_moe_config(router_sliding_windows=16)
        self.assertEqual(config.router_sliding_window, 16)
        self.assertEqual(config.router_sliding_windows, 16)

    def test_model_forward(self):
        config = get_pangu_v2_moe_config()
        model = PanguV2MoEForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = self.get_input_ids(vocab_size=config.vocab_size)
        attention_mask = torch.ones_like(input_ids, device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

        self.assertEqual(outputs.logits.shape, (2, 5, config.vocab_size))
        self.assertIsNotNone(outputs.loss)

    def test_output_router_logits(self):
        config = get_pangu_v2_moe_config(output_router_logits=True)
        model = PanguV2MoEForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = self.get_input_ids(vocab_size=config.vocab_size)

        with torch.no_grad():
            outputs = model(input_ids, output_router_logits=True)

        self.assertEqual(len(outputs.router_logits), config.num_hidden_layers - config.first_k_dense_replace)
        for router_logits in outputs.router_logits:
            self.assertEqual(router_logits.shape, (input_ids.numel(), config.n_routed_experts))
        self.assertIsNotNone(outputs.aux_loss)

    def test_cache_with_param_sink(self):
        config = get_pangu_v2_moe_config()
        model = PanguV2MoEForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = self.get_input_ids(batch_size=1, seq_length=4, vocab_size=config.vocab_size)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            next_outputs = model(
                input_ids[:, -1:],
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )

        self.assertEqual(outputs.logits.shape, (1, 4, config.vocab_size))
        self.assertEqual(next_outputs.logits.shape, (1, 1, config.vocab_size))
        self.assertEqual(next_outputs.past_key_values.get_seq_length(), config.param_sink_number + 5)

    def test_generate(self):
        config = get_pangu_v2_moe_config()
        model = PanguV2MoEForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = self.get_input_ids(batch_size=1, seq_length=3, vocab_size=config.vocab_size)

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)

        self.assertEqual(generated.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
