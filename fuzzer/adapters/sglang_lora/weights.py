"""Random LoRA weight generation for SGLang LoRA fuzzing.

Delegates to the shared implementation in vllm_lora.weights — the weight
generation utility has no backend-specific dependencies.
"""
from ...adapters.vllm_lora.weights import generate_random_lora  # noqa: F401
