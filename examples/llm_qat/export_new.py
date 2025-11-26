# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_hf_checkpoint
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.quantization.utils import set_quantizer_state_dict
from modelopt.torch.utils import print_rank_0

RAND_SEED = 1234

# Enable automatic save/load of modelopt state huggingface checkpointing
mto.enable_huggingface_checkpointing()


def get_model(
    ckpt_path: str,
    device="cuda",
):
    """
    Loads a QLoRA model that has been trained using modelopt trainer.
    """
    # TODO: Add support for merging adapters in BF16 and merging adapters with quantization for deployment
    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map)

    # Try to load modelopt_state.pt if it exists (e.g. from FSDP training)
    modelopt_state_path = Path(ckpt_path) / "modelopt_state.pt"
    if modelopt_state_path.exists():
        print_rank_0(f"Loading modelopt state from {modelopt_state_path}...")
        modelopt_state = torch.load(modelopt_state_path, weights_only=False, map_location="cpu")
        
        try:
            restore_from_modelopt_state(model, modelopt_state)
            print_rank_0("Restored modelopt state from modelopt_state.pt")
        except Exception as e:
            if "Model already has modelopt state" in str(e):
                print_rank_0("Model already has modelopt state, manually loading quantizer statistics...")
                # Manual loading of quantizer state
                # structure: modelopt_state['modelopt_state_dict'] -> list -> tuple -> dict -> metadata -> quantizer_state
                if "modelopt_state_dict" in modelopt_state:
                    state_list = modelopt_state["modelopt_state_dict"]
                    if isinstance(state_list, list) and len(state_list) > 0:
                        # We assume the first mode is quantization
                        _, mode_data = state_list[0]
                        if "metadata" in mode_data and "quantizer_state" in mode_data["metadata"]:
                            quantizer_state = mode_data["metadata"]["quantizer_state"]
                            print_rank_0(f"Found {len(quantizer_state)} quantizer parameters. Loading...")
                            
                            # load_state_dict with strict=False to ignore other model weights
                            missing, unexpected = model.load_state_dict(quantizer_state, strict=False)
                            print_rank_0(f"Loaded quantizer state. Missing keys (subset): {missing[:5]}")
                        else:
                            print_rank_0("Warning: Could not find quantizer_state in modelopt_state metadata.")
            else:
                raise e

    # Restore modelopt state for LoRA models. For QAT/QAD models from_pretrained call handles this
    if hasattr(model, "peft_config"):
        modelopt_state = torch.load(f"{ckpt_path}/modelopt_state_train.pth", weights_only=False)
        restore_from_modelopt_state(model, modelopt_state)
        print_rank_0("Restored modelopt state")

        # Restore modelopt quantizer state dict
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        if modelopt_weights is not None:
            set_quantizer_state_dict(model, modelopt_weights)
            print_rank_0("Restored modelopt quantizer state dict")

    return model


def main(args):
    # Load model
    model = get_model(args.pyt_ckpt_path, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pyt_ckpt_path)
    is_qlora = hasattr(model, "peft_config")

    # Export HF checkpoint
    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    if is_qlora:
        base_model_dir = export_dir / "base_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_model_dir = export_dir

    try:
        post_state_dict, hf_quant_config = _export_hf_checkpoint(model, is_modelopt_qlora=is_qlora)

        with open(f"{base_model_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        if is_qlora:
            model.base_model.save_pretrained(f"{base_model_dir}", state_dict=post_state_dict)
            model.save_pretrained(export_dir)
        else:
            model.save_pretrained(export_dir, state_dict=post_state_dict)

        config_path = f"{base_model_dir}/config.json"

        config_data = model.config.to_dict()

        config_data["quantization_config"] = hf_quant_config

        with open(config_path, "w") as file:
            json.dump(config_data, file, indent=4)

        # Save tokenizer
        tokenizer.save_pretrained(export_dir)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        default="/apps/quant_models/Qwen3-8B-Base-int4-awq-blockwise-qat/checkpoint-32",
        help="Specify where the PyTorch checkpoint path is",
    )

    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--export_path",
        default="/apps/quant_models/Qwen3-8B-Base-int4-awq-blockwise-qat/hf_checkpoint",
        help="Path to save the exported model",
    )

    args = parser.parse_args()

    main(args)


# python3 export.py --pyt_ckpt_path qwen3_8b_qat_fsdp --export_path qwen3_8b_qat_exported
# python3 export.py --pyt_ckpt_path qwen3_0.6b_qat --export_path qwen3_0.6b_qat_exported

# python3 export.py --pyt_ckpt_path qwen3_0.6b_qat_awq_full_fsdp --export_path qwen3_0.6b_qat_awq_full_fsdp_exported
# python3 export.py --pyt_ckpt_path qwen3_0.6b_qat_fsdp --export_path qwen3_0.6b_qat_fsdp_exported