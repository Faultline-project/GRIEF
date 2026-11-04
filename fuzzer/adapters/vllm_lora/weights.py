"""
Random LoRA weight generation utility.

Two adapters initialised with different seeds produce maximally different
outputs from the same base model, making cross-adapter KV contamination
immediately visible in the oracle.

Requires: pip install peft transformers torch
"""

from __future__ import annotations

import os


def generate_random_lora(
    base_model_name: str,
    output_dir: str,
    adapter_name: str,
    rank: int = 8,
    seed: int = 42,
) -> str:
    """
    Generate a random LoRA adapter via PEFT and save to disk.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model ID or local path (e.g. "Qwen/Qwen2.5-0.5B-Instruct").
    output_dir : str
        Root directory under which adapter_name/ will be created.
    adapter_name : str
        Subdirectory name for this adapter (e.g. "lora_a").
    rank : int
        LoRA rank r. Higher = more parameters changed. 8 is a good default.
    seed : int
        RNG seed. Different seeds → different weight tensors → different outputs.

    Returns
    -------
    str
        Absolute path to the saved adapter directory.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    # PEFT initializes lora_B to zeros so that delta_W = B @ A = 0 at init.
    # That is correct for training (adapter starts as identity) but wrong for
    # fuzzing: zero-delta adapters are indistinguishable from the base model,
    # making cross-adapter contamination undetectable.
    # Re-initialize B with a small random normal so each adapter produces
    # genuinely different outputs and contamination is immediately visible.
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)

    out_path = os.path.join(output_dir, adapter_name)
    os.makedirs(out_path, exist_ok=True)
    peft_model.save_pretrained(out_path)
    return out_path
