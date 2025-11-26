#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple QAT + FSDP Training Example

测试 ModelOpt QAT 在 FSDP 环境下是否可行

支持的模型:
    - Llama 系列 (Llama-2, Llama-3)
    - Qwen2 系列
    - Qwen3 系列 (使用与 Qwen2 相同的架构)
    - Mistral 系列

Usage:
    # Llama 模型
    torchrun --nproc_per_node=2 simple_qat_fsdp_train.py --model-path meta-llama/Llama-3.2-1B
    
    # Qwen2 模型
    torchrun --nproc_per_node=2 simple_qat_fsdp_train.py --model-path Qwen/Qwen2-0.5B --output-dir qwen2_qat_output
    
    # Qwen3 模型 (自动检测，无需手动指定 layer)
    torchrun --nproc_per_node=2 simple_qat_fsdp_train.py --model-path Qwen/Qwen3-0.6B --output-dir qwen3_qat_output
    
    # 或使用 accelerate
    accelerate launch --num_processes 2 simple_qat_fsdp_train.py --model-path meta-llama/Llama-3.2-1B
"""

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils import get_daring_anteater

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def setup_distributed():
    """初始化分布式环境"""
    if not dist.is_initialized():
        # 尝试从环境变量获取
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        elif "LOCAL_RANK" in os.environ:
            # torchrun 方式
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
            world_size = 1
            local_rank = 0
        
        if world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return rank, world_size, local_rank


def get_dataloader(args, tokenizer, rank, world_size):
    """创建分布式数据加载器"""
    train_dataset = get_daring_anteater(
        tokenizer, "train", args.max_length, args.train_size, args.calib_size
    )
    calib_dataset = get_daring_anteater(
        tokenizer, "test", args.max_length, args.train_size, args.calib_size
    )
    
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }
    
    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    calib_sampler = DistributedSampler(
        calib_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        sampler=calib_sampler,
        collate_fn=collate_fn,
    )
    
    return train_dataloader, calib_dataloader


def train(model, optimizer, train_dataloader, tokenizer, epochs, output_dir, rank, world_size):
    """训练循环"""
    model.train()
    
    for epoch in range(epochs):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")
        
        # 设置 sampler 的 epoch（重要！）
        train_dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") if rank == 0 else train_dataloader
        
        for batch_idx, batch in enumerate(iterator):
            inputs = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            # 前向传播
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=inputs
            )
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if rank == 0 and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 定期打印
            if rank == 0 and batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        if rank == 0:
            print(f"Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")


def save_fsdp_model(model, tokenizer, output_dir, rank, world_size):
    """保存 FSDP 模型 - 包含训练后的 quantizer 参数"""
    # FSDP 需要所有 rank 参与 state_dict 获取，不能提前 return
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print("Collecting full model state_dict from FSDP...")
    
    # 获取完整的 state_dict（包括 quantizer 参数）
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state_dict = model.state_dict()
        
        if rank == 0:
            # 分离普通权重和 quantizer 权重
            model_weights = {}
            quantizer_weights = {}
            
            for k, v in full_state_dict.items():
                # quantizer 相关参数单独保存
                if any(x in k for x in ['quantizer', '_amax', '_pre_quant_scale']):
                    quantizer_weights[k] = v
                else:
                    model_weights[k] = v
            
            print(f"Saving model weights ({len(model_weights)} parameters)...")
            print(f"Extracting quantizer weights ({len(quantizer_weights)} parameters)...")
            
            # 保存模型权重（不包含 quantizer）
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model.save_pretrained(
                output_dir,
                state_dict=model_weights,
                safe_serialization=False,
            )
            
            # 保存 tokenizer
            print("Saving tokenizer...")
            tokenizer.save_pretrained(output_dir)
            
            # 保存 ModelOpt 状态 - 训练后的状态
            print("Saving ModelOpt state with trained quantizer parameters...")
            # 从当前模型提取 modelopt_state，但替换其中的 quantizer_state 为训练后的值
            try:
                # 获取当前的 modelopt state 结构
                current_modelopt_state = mto.modelopt_state(unwrapped_model)
                
                # 更新 quantizer_state 为训练后的参数
                if "modelopt_state_dict" in current_modelopt_state:
                    state_list = current_modelopt_state["modelopt_state_dict"]
                    if isinstance(state_list, list) and len(state_list) > 0:
                        for i, (mode_name, mode_data) in enumerate(state_list):
                            if "metadata" in mode_data and "quantizer_state" in mode_data["metadata"]:
                                # 替换为训练后的 quantizer 参数
                                state_list[i] = (mode_name, {
                                    **mode_data,
                                    "metadata": {
                                        **mode_data["metadata"],
                                        "quantizer_state": quantizer_weights
                                    }
                                })
                                print(f"✓ Updated quantizer_state with {len(quantizer_weights)} trained parameters")
                
                # 保存更新后的 modelopt_state
                torch.save(current_modelopt_state, os.path.join(output_dir, "modelopt_state.pt"))
                print("✓ ModelOpt state saved successfully!")
                
            except Exception as e:
                print(f"Warning: Failed to construct modelopt_state: {e}")
                print("Falling back to saving quantizer weights directly as modelopt_state.pth...")
                # 备用方案：直接保存 quantizer 权重（兼容原版格式）
                torch.save(quantizer_weights, os.path.join(output_dir, "modelopt_state.pth"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple QAT + FSDP Training Script")
    
    # Model and data
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--train-size", type=int, default=512, help="Train size")
    parser.add_argument("--calib-size", type=int, default=0, help="Calibration size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    # Quantization config
    parser.add_argument(
        "--quant-cfg",
        type=str,
        default="NVFP4_DEFAULT_CFG",
        choices=list(mtq.config.choices) + [
            "NVFP4_DEFAULT_BS32_CFG",
            "NVFP4_DEFAULT_BS64_CFG",
            "NVFP4_SELECTIVE_CFG",
            "NVFP4_ATTN_ONLY_CFG",
            "NVFP4_HYBRID_CFG",
        ],
        help="Quantization configuration (including custom configs)",
    )
    
    # FSDP config
    parser.add_argument("--use-fsdp", action="store_true", default=True, help="Use FSDP")
    parser.add_argument(
        "--fsdp-transformer-layer",
        type=str,
        default=None,
        help="Transformer layer class name for FSDP auto wrap (e.g., LlamaDecoderLayer, Qwen2DecoderLayer, Qwen3DecoderLayer, MistralDecoderLayer). If None, will auto-detect."
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="qat_fsdp_output", help="Output directory")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("="*70)
        print("ModelOpt QAT + FSDP Training")
        print("="*70)
        print(f"World Size: {world_size}")
        print(f"Model: {args.model_path}")
        print(f"Quantization: {args.quant_cfg}")
        print(f"Batch Size: {args.batch_size} per device")
        print(f"Epochs: {args.epochs}")
        print(f"Output: {args.output_dir}")
        print("="*70)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 1. 加载模型和 tokenizer
    if rank == 0:
        print("\n[1/5] Loading model and tokenizer...")
    
    # 启用 HuggingFace checkpointing
    mto.enable_huggingface_checkpointing()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.cuda()
    
    # 2. 准备数据
    if rank == 0:
        print("\n[2/5] Preparing data loaders...")
    
    train_dataloader, calib_dataloader = get_dataloader(args, tokenizer, rank, world_size)
    
    # 3. 量化模型（在 FSDP 之前！）
    if rank == 0:
        print(f"\n[3/5] Quantizing model with {args.quant_cfg}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 校准模型（与非 FSDP 版本对齐）
    def calibrate(m: nn.Module):
        for batch in calib_dataloader:
            m(batch["input_ids"].to(device))
    
    # 量化模型
    # 检查是否使用自定义配置
    custom_configs = [
        "NVFP4_DEFAULT_BS32_CFG",
        "NVFP4_DEFAULT_BS64_CFG",
        "NVFP4_SELECTIVE_CFG",
        "NVFP4_ATTN_ONLY_CFG",
        "NVFP4_HYBRID_CFG",
    ]
    
    if args.quant_cfg in custom_configs:
        # 导入自定义配置
        from custom_nvfp4_configs import (
            NVFP4_DEFAULT_BS32_CFG,
            NVFP4_DEFAULT_BS64_CFG,
            NVFP4_SELECTIVE_CFG,
            NVFP4_ATTN_ONLY_CFG,
            NVFP4_HYBRID_CFG,
        )
        quant_cfg = locals()[args.quant_cfg]
        if rank == 0:
            print(f"✓ 使用自定义配置: {args.quant_cfg}")
    else:
        # 使用标准配置
        quant_cfg = getattr(mtq, args.quant_cfg)
    
    model = mtq.quantize(model, quant_cfg, calibrate)
    
    if rank == 0:
        print("✓ Model quantized successfully!")
        quantized_layers = sum(1 for name, module in model.named_modules() 
                              if hasattr(module, 'weight_quantizer'))
        print(f"✓ Number of quantized layers: {quantized_layers}")
    
    # 4. 应用 FSDP（在量化之后！）
    if args.use_fsdp and world_size > 1:
        if rank == 0:
            print(f"\n[4/5] Wrapping model with FSDP...")
        
        # 获取 transformer layer class
        def get_transformer_layer_cls(model_path, layer_name=None):
            """自动检测或根据名称获取 transformer layer class
            
            支持的模型类型：
            - Llama (llama, llama2, llama3)
            - Qwen2 (qwen2 使用 Qwen2DecoderLayer)
            - Qwen3 (qwen3 使用 Qwen3DecoderLayer) - 注意：Qwen3 有独立的 DecoderLayer！
            - Mistral
            """
            config = AutoConfig.from_pretrained(model_path)
            model_type = config.model_type
            
            # 如果指定了 layer name，直接导入
            if layer_name:
                if layer_name == "LlamaDecoderLayer":
                    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                    return LlamaDecoderLayer
                elif layer_name == "Qwen2DecoderLayer":
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                    return Qwen2DecoderLayer
                elif layer_name == "Qwen3DecoderLayer":
                    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
                    return Qwen3DecoderLayer
                elif layer_name == "MistralDecoderLayer":
                    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                    return MistralDecoderLayer
                else:
                    raise ValueError(f"Unknown layer name: {layer_name}")
            
            # 否则根据 model_type 自动检测
            if model_type == "llama":
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                return LlamaDecoderLayer
            elif model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                return Qwen2DecoderLayer
            elif model_type == "qwen3":
                # Qwen3 有独立的 DecoderLayer 类（不同于 Qwen2）
                from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
                return Qwen3DecoderLayer
            elif model_type == "mistral":
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                return MistralDecoderLayer
            else:
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: llama, qwen2, qwen3, mistral. "
                    f"Please specify --fsdp-transformer-layer explicitly."
                )
        
        transformer_layer_cls = get_transformer_layer_cls(
            args.model_path, 
            args.fsdp_transformer_layer
        )
        
        if rank == 0:
            print(f"  Using transformer layer: {transformer_layer_cls.__name__}")
        
        # FSDP auto wrap policy
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
        
        # 包装模型
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            mixed_precision=None,  # 使用 bf16
        )
        
        if rank == 0:
            print("✓ FSDP wrapper applied successfully!")
    else:
        if rank == 0:
            print(f"\n[4/5] Skipping FSDP (world_size={world_size})")
    
    # 5. 训练
    if rank == 0:
        print(f"\n[5/5] Starting QAT training...")
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 训练
    train(model, optimizer, train_dataloader, tokenizer, args.epochs, args.output_dir, rank, world_size)
    
    # 训练完成后保存（包含训练后的 quantizer 参数）
    if args.output_dir:
        if rank == 0:
            print(f"\nSaving model to {args.output_dir}...")
        save_fsdp_model(model, tokenizer, args.output_dir, rank, world_size)
    
    if rank == 0:
        print("\n" + "="*70)
        print("Training completed successfully!")
        print(f"Model saved to: {args.output_dir}")
        print("="*70)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# 使用示例（会自动检测 DecoderLayer 类型，无需指定 --fsdp-transformer-layer）：

# Qwen3 模型 - 自动检测使用 Qwen3DecoderLayer
# torchrun --nproc_per_node=4 simple_qat_fsdp_train.py --use-fsdp --quant-cfg NVFP4_DEFAULT_CFG --model-path Qwen/Qwen3-8B --output-dir qwen3_8b_qat_fsdp
# torchrun --nproc_per_node=4 simple_qat_fsdp_train.py --use-fsdp --quant-cfg NVFP4_DEFAULT_CFG --model-path qQwen/Qwen3-0.6B --output-dir qwen3_0.6b_qat_fsdp 

# Qwen3 + AWQ 配置（推荐用于 fuse 优化）
# torchrun --nproc_per_node=4 --master-port 29501 simple_qat_fsdp_train.py --use-fsdp --quant-cfg NVFP4_AWQ_FULL_CFG --model-path Qwen/Qwen3-0.6B --output-dir qwen3_0.6b_qat_awq_full_fsdp


# 可用的量化配置：
# "NVFP4_DEFAULT_CFG"       - 基础配置（无 AWQ，不会 fuse）
# "NVFP4_AWQ_LITE_CFG"      - AWQ Lite（会 fuse）
# "NVFP4_AWQ_CLIP_CFG"      - AWQ Clip（会 fuse）
# "NVFP4_AWQ_FULL_CFG"      - AWQ Full（会 fuse，推荐）
# "NVFP4_AFFINE_KV_CFG"     - KV Cache 量化
# "NVFP4_FP8_MHA_CONFIG"    - FP8 MHA
