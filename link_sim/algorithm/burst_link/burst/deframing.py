"""单突发解帧：相位跟踪、软解调和信道译码。"""
from __future__ import annotations

import torch

from link_sim.algorithm.burst_link.coding.decoder import create_decoder
from link_sim.algorithm.burst_link.modulation.soft import soft_demodulate
from link_sim.algorithm.burst_link.phase_tracking import PilotPhaseTracker
from .framing import BurstFramer


class BurstDeframer:
    """按组帧配置恢复整个突发的信息块，不裁切有效信息长度。

    输入是对齐同步头首符号的一维复符号，长度必须等于组帧总符号数。
    输出是一维 uint8 比特张量，长度等于组帧信息容量，包含信息补零。
    相位轨迹可在成功调用后通过 last_phase_result 查看。对象独立处理
    每个突发，不读取发射比特或信道真值。默认使用所有帧头估计
    幅度归一化后的复符号等效误差功率，作为软解调的噪声功率。
    可选 noise_variance 手动指定，单位为每复符号噪声功率。
    last_estimated_noise_variance 和 last_noise_variance 分别保存
    最近一次估计值和实际用于软解调的值。
    """
    def __init__(self, framer: BurstFramer, *, noise_variance: float | None = None,
                 num_iter: int = 20, phase_tracker=None):
        if not isinstance(framer, BurstFramer):
            raise TypeError("framer 必须是 BurstFramer 对象")
        self.framer = framer
        if isinstance(noise_variance, bool):
            raise TypeError("noise_variance 必须是有限正数或 None")
        self.noise_variance = None if noise_variance is None else float(noise_variance)
        if self.noise_variance is not None and not 0 < self.noise_variance < float("inf"):
            raise ValueError("noise_variance 必须是有限正数或 None")
        self.phase_tracker = phase_tracker if phase_tracker is not None else PilotPhaseTracker(sync_symbols=framer.sync_symbols)
        if not callable(self.phase_tracker):
            raise TypeError("phase_tracker 必须可调用")
        self.decoder = create_decoder(framer.encoder, num_iter=num_iter)
        self.last_phase_result = None
        self.last_estimated_noise_variance = None
        self.last_noise_variance = None

    @torch.no_grad()
    def __call__(self, symbols: torch.Tensor) -> torch.Tensor:
        """输入完整突发复符号，输出含补零的一维信息位。"""
        self.last_phase_result = None
        self.last_estimated_noise_variance = None
        self.last_noise_variance = None
        if not isinstance(symbols, torch.Tensor) or symbols.ndim != 1 or not symbols.is_complex():
            raise TypeError("symbols 必须是一维复数张量")
        if symbols.numel() != self.framer.total_symbols:
            raise ValueError(f"输入突发符号数必须为 {self.framer.total_symbols}")
        if symbols.device != self.framer.device:
            raise ValueError("输入符号设备必须与组帧配置一致")
        tracked = self.phase_tracker(symbols, header_spans=self.framer.header_spans,
                                     header_symbols=self.framer.header_symbols,
                                     modulation=self.framer.data_modulation)
        # 所有帧头均已归一化和相位补偿，残差功率的单位与软解调输入一致。
        # 这包含残留信道失真及相位误差，不等同于已知的纯加性噪声。
        reference = self.framer.header_symbols.to(dtype=tracked.symbols.dtype)
        if reference.numel() < 2 and self.noise_variance is None:
            raise ValueError("帧头少于 2 个符号，无法估计误差功率；请手动指定 noise_variance")
        header_residuals = [tracked.symbols[start:end] - reference
                            for start, end in self.framer.header_spans]
        estimated_variance = float(torch.cat(header_residuals).abs().square().mean().item())
        if not 0 <= estimated_variance < float("inf"):
            raise ValueError("帧头等效误差功率无效")
        # 全理想输入可能产生零残差；避免 LLR 除零，保留原始估计用于查看。
        used_variance = max(estimated_variance, 1e-8) if self.noise_variance is None else self.noise_variance
        spans = [span for group in self.framer.codeword_spans for span in group]
        if not spans:
            result = torch.empty(0, dtype=torch.uint8, device=symbols.device)
        else:
            data = torch.stack([tracked.symbols[start:end] for start, end in spans])
            logits = soft_demodulate(data, self.framer.data_modulation,
                                     noise_variance=used_variance)
            logits = logits[..., :self.framer.encoded_bits_per_codeword]
            words = self.decoder.decode(logits)
            expected = (self.framer.codeword_count, self.framer.information_bits_per_codeword)
            if tuple(words.shape) != expected:
                raise RuntimeError(f"译码输出尺寸必须为 {expected}")
            result = words.reshape(-1).to(torch.uint8)
        self.last_phase_result = tracked
        self.last_estimated_noise_variance = estimated_variance
        self.last_noise_variance = used_variance
        return result
