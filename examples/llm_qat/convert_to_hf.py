#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simple script to convert a trained modelopt checkpoint to HuggingFace format

import argparse
import os

import torch
import transformers

import modelopt.torch.opt as mto
from modelopt.torch.export import export_hf_checkpoint

# Enable automatic save/load of modelopt state
mto.enable_huggingface_checkpointing()


def main():
    parser = argparse.ArgumentParser(
        description="Convert modelopt checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/apps/quant_models/Qwen3-8B-Base-Int4-blockwise-qat/checkpoint-128",
        help="Directory containing the trained model checkpoint with modelopt_state",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/apps/quant_models/Qwen3-8B-Base-Int4-blockwise-qat/hf_checkpoint",
        help="Directory to save the exported HuggingFace checkpoint",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for loading the model",
    )
    
    args = parser.parse_args()
    
    # Determine the model path
    model_path = args.checkpoint_dir
    
    # Set torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    print(f"Loading model from: {model_path}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Load the model with modelopt state
    # If the checkpoint_dir contains the model, HuggingFace will automatically load modelopt_state
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch_dtype,
    )
    
    print(f"Model loaded successfully")
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_dir)
    
    print(f"Exporting to HuggingFace checkpoint format...")
    
    # Export to HuggingFace checkpoint
    with torch.inference_mode():
        export_hf_checkpoint(
            model,
            export_dir=args.output_dir,
        )
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ Successfully exported HuggingFace checkpoint to: {args.output_dir}")
    print(f"✓ Tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

