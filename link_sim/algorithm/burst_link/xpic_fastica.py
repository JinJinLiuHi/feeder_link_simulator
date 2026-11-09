"""复数 FastICA：对共同采样时基上的两路复基带波形做瞬时盲分离。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch


@dataclass(frozen=True)
class FastICAResult:
    """分离波形和诊断；输出顺序、相位及幅度均为 ICA 不确定量。

    ``unmixing`` 采用行样本约定：``components = (samples - mean) @ unmixing``。
    ``condition_number`` 是训练窗口接收数据的协方差条件数平方根，
    并非真实信道条件数或分离成功的保证。
    """
    components: tuple[torch.Tensor, torch.Tensor]
    unmixing: torch.Tensor
    mean: torch.Tensor
    iterations: int
    converged: bool
    convergence_error: float
    condition_number: float
    excess_kurtosis: torch.Tensor


class ComplexFastICA:
    """双通道、复数、瞬时混合模型的对称 FastICA。

    训练窗口只用于估计盲分离矩阵，不读取业务真值或头部序列，随后将
    该矩阵应用于完整输入波形；输入保持同一采样时间轴。假设训练窗口内
    两个源近似统计独立、非高斯，且混合矩阵基本不变且满秩。该单抽头
    模型不能保证分离差分时延或频率选择性导致的卷积混合。

    参考 Bingham & Hyvärinen, Int. J. Neural Systems 10(1), 2000,
    doi:10.1142/S0129065700000028，式 (13) 和 (15)：对白化复数数据采用
    g(u)=1/(1+u) 的固定点更新及对称去相关。此为 PyTorch 独立实现。
    ICA 收敛只表明迭代变化小，不保证独立性假设满足或分离成功。
    """

    def __init__(self, *, max_iter: int = 500, tolerance: float = 1e-6,
                 max_condition_number: float = 1e4):
        if isinstance(max_iter, bool):
            raise TypeError("max_iter 必须是正整数")
        try:
            self.max_iter = operator.index(max_iter)
        except TypeError as exc:
            raise TypeError("max_iter 必须是正整数") from exc
        if self.max_iter < 1:
            raise ValueError("max_iter 必须是正整数")
        self.tolerance = float(tolerance)
        self.max_condition_number = float(max_condition_number)
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("tolerance 必须为 0 到 1 之间的有限数")
        if not math.isfinite(self.max_condition_number) or self.max_condition_number <= 1:
            raise ValueError("max_condition_number 必须为大于 1 的有限数")

    @staticmethod
    def _orthogonalize(weights: torch.Tensor) -> torch.Tensor:
        gram = weights.mH @ weights
        eig, vec = torch.linalg.eigh(gram)
        floor = torch.finfo(eig.dtype).eps * eig[-1] * 2
        if not bool((eig[-1] > torch.finfo(eig.dtype).tiny).item()) or not bool((eig[0] > floor).item()):
            raise ValueError("FastICA 迭代方向退化，不能分离两路信号")
        inv_sqrt = (vec * eig.rsqrt().unsqueeze(0)) @ vec.mH
        return weights @ inv_sqrt

    @torch.no_grad()
    def __call__(self, waveform_a: torch.Tensor, waveform_b: torch.Tensor, *,
                 start: int, end: int, training_mask: torch.Tensor | None = None) -> FastICAResult:
        """在 [start, end) 中用可选掩码选择训练样本，输出保留完整时间轴。"""
        inputs = (waveform_a, waveform_b)
        if any(not isinstance(x, torch.Tensor) or x.ndim != 1 for x in inputs):
            raise TypeError("输入必须是两路一维 torch.Tensor")
        if waveform_a.dtype not in (torch.complex64, torch.complex128) or waveform_b.dtype != waveform_a.dtype:
            raise TypeError("两路必须具有相同的 complex64 或 complex128 精度")
        if waveform_a.device != waveform_b.device:
            raise ValueError("两路波形必须在同一设备")
        if waveform_a.numel() != waveform_b.numel():
            raise ValueError("两路波形必须长度相同且处于共同采样时基")
        if any(not bool(torch.isfinite(x).all().item()) for x in inputs):
            raise ValueError("输入不得包含非有限值")
        if isinstance(start, bool) or isinstance(end, bool):
            raise TypeError("训练区间坐标必须是整数")
        try:
            start, end = operator.index(start), operator.index(end)
        except TypeError as exc:
            raise TypeError("训练区间坐标必须是整数") from exc
        if not 0 <= start < end <= waveform_a.numel() or end - start < 128:
            raise ValueError("训练区间须位于波形内且至少包含 128 个采样")

        samples = torch.stack(inputs, dim=1)
        segment = samples[start:end]
        if training_mask is not None:
            if (not isinstance(training_mask, torch.Tensor) or training_mask.dtype != torch.bool
                    or training_mask.shape != waveform_a.shape or training_mask.device != waveform_a.device):
                raise ValueError("训练掩码须与输入波形等长、同设备且为布尔类型")
            segment = segment[training_mask[start:end]]
            if segment.shape[0] < 128:
                raise ValueError("FastICA 训练窗口内有效业务样本不足 128 点，请增加训练窗口或业务符号数")
        mean = segment.mean(dim=0)
        centered = segment - mean
        covariance = centered.mH @ centered / centered.shape[0]
        eig, vec = torch.linalg.eigh(covariance)
        if not bool(torch.isfinite(eig).all().item()) or not bool((eig[-1] > 0).item()):
            raise ValueError("训练窗口没有足够有效信号能量")
        floor = torch.finfo(eig.dtype).eps * eig[-1] * 2
        if not bool((eig[0] > floor).item()):
            raise ValueError("接收数据协方差退化，无法分离两路")
        condition = float(torch.sqrt(eig[-1] / eig[0]).item())
        if condition > self.max_condition_number:
            raise ValueError("接收数据协方差条件数过大，无法稳定白化")
        whitening = (vec * eig.rsqrt().unsqueeze(0)) @ vec.mH
        z = centered @ whitening

        # 与白化基底不对称的确定性初值；不改变调用方随机数状态。
        initial = torch.tensor([[1.0, 0.23 + 0.37j], [-0.23 + 0.37j, 1.0]],
                               dtype=z.dtype, device=z.device)
        weights = self._orthogonalize(initial)
        converged = False
        delta = float("inf")
        for iteration in range(1, self.max_iter + 1):
            y = z @ weights
            u = y.abs().square()
            g = (1 + u).reciprocal()
            derivative = -(g.square())
            correction = (g + u * derivative).mean(dim=0)
            update = (z.mH @ (y * g)) / z.shape[0] - weights * correction.unsqueeze(0)
            new_weights = self._orthogonalize(update)
            alignment = (weights.conj() * new_weights).sum(dim=0).abs()
            delta = float((1 - alignment).abs().max().item())
            weights = new_weights
            if delta <= self.tolerance:
                converged = True
                break
        unmixing = whitening @ weights
        separated = (samples - mean) @ unmixing
        fit = (segment-mean) @ unmixing
        power = fit.abs().square().mean(dim=0)
        pseudo = fit.square().mean(dim=0)
        kurtosis = (fit.abs().pow(4).mean(dim=0) / power.square()
                    - 2 - pseudo.abs().square() / power.square())
        return FastICAResult((separated[:, 0].contiguous(), separated[:, 1].contiguous()),
                             unmixing, mean, iteration, converged, delta,
                             condition, kurtosis)
