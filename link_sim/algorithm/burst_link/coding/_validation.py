from __future__ import annotations

import torch


def validate_information_words(words: torch.Tensor, k: int, device: torch.device) -> None:
    """Validate a single information block or an array of blocks on the encoder's device."""
    if not isinstance(words, torch.Tensor):
        raise TypeError("信息块必须是 PyTorch Tensor")
    if words.ndim < 1 or words.shape[-1] != k:
        raise ValueError(f"信息块末维应为 {k}，实际形状为 {tuple(words.shape)}")
    if words.device != device:
        raise ValueError(f"信息块位于 {words.device}，编码器位于 {device}；请显式选择相同设备")
    if words.is_complex():
        raise ValueError("信息块必须为二进制实数或整数")
    if not bool(torch.all((words == 0) | (words == 1))):
        raise ValueError("信息块只能包含 0 和 1")


def resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 环境不可使用 CUDA")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved
