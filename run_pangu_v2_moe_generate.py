#!/usr/bin/env python
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Run generation with a local Pangu V2 MoE checkpoint."""

import argparse
import json
import logging

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with PanguV2MoEForCausalLM real weights.")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to the local Pangu V2 MoE checkpoint directory, or a Hub id if available.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Tokenizer path. Defaults to --model_name_or_path.",
    )
    parser.add_argument("--prompt", default="你好，请介绍一下盘古大模型。", help="Prompt text to generate from.")
    parser.add_argument(
        "--input_ids",
        default=None,
        help="Comma-separated token ids. If set, tokenizer loading and prompt tokenization are skipped.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device_map passed to from_pretrained, for example 'auto', 'cpu', or 'cuda:0'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used when --device_map is not set. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_input_ids", action="store_true")
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype):
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def build_inputs(args, tokenizer):
    if args.input_ids is not None:
        input_ids = [int(token_id.strip()) for token_id in args.input_ids.split(",") if token_id.strip()]
        return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}

    encoded = tokenizer(args.prompt, return_tensors="pt")
    if "token_type_ids" in encoded:
        encoded.pop("token_type_ids")
    return encoded


def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if config.model_type != "pangu_v2_moe":
        logger.warning("Loaded config model_type=%s, expected pangu_v2_moe.", config.model_type)
    if args.print_config:
        logger.info("Config:\n%s", json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    tokenizer = None
    if args.input_ids is None:
        tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": resolve_dtype(args.dtype),
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()

    inputs = build_inputs(args, tokenizer)
    if args.print_input_ids:
        logger.info("Input ids: %s", inputs["input_ids"].tolist())

    if args.device_map is None:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        first_param = next(model.parameters())
        inputs = {key: value.to(first_param.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "use_cache": args.use_cache,
    }
    if args.min_new_tokens is not None:
        generation_kwargs["min_new_tokens"] = args.min_new_tokens
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else config.eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else config.pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    print("Generated token ids:")
    print(output_ids[0].tolist())

    if tokenizer is not None:
        print("\nDecoded text:")
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
