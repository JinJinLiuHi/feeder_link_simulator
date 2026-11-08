"""对一维复数波形在指定采样位置执行四点三次插值。"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


class FractionalInterpolator:
    """按采样点坐标提取等间隔的复数样值。

    输入是在同一采样时间轴上的一维 complex64 或 complex128 张量。
    对每个目标位置 t = start + k*step，取 floor(t) 左一和右二共四点，
    采用三次 Lagrange 多项式；系数可写为 Farrow 结构。对象不估计
    定时，不进行自动同步，也不假定输出的 step 必须等于每符号采样数。
    """

    def __call__(self, waveform: torch.Tensor, *, start: float, step: float,
                 count: int) -> torch.Tensor:
        """返回长度为 count、设备和复数精度与 waveform 相同的一维张量。

        start: 第一个目标位置，相对于输入张量第 0 个采样，以采样点为单位；
        step: 相邻目标位置间隔，以输入采样点为单位，必须为有限正数；
        count: 要生成的采样个数，必须为非负整数。

        每个目标位置 t 均须满足 1 <= floor(t) 且 floor(t)+2 < N。
        即使 t 为整数也保留同样的四点边界约束，不自动截断或补零。
        count=0 时返回空张量，不要求 start 位于输入范围内。
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform 必须是 PyTorch 复数张量")
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("waveform 必须是非空的一维复数张量")
        if waveform.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("waveform 必须为 complex64 或 complex128")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("waveform 不能包含非有限值")
        if isinstance(start, bool) or not isinstance(start, Real) or not math.isfinite(float(start)):
            raise ValueError("start 必须是有限实数，单位为输入采样")
        if isinstance(step, bool) or not isinstance(step, Real) or not math.isfinite(float(step)) or float(step) <= 0:
            raise ValueError("step 必须是有限正数，单位为输入采样")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("count 必须是非负整数")

        start, step, count = float(start), float(step), int(count)
        if count == 0:
            return waveform.new_empty((0,))

        last = start + (count - 1) * step
        if (not math.isfinite(last) or start < 1 or
                math.floor(last) + 2 >= waveform.numel()):
            raise ValueError("插值位置越界：每个位置须保留左 1 点及右 2 点")

        # 使用 float64 位置计算避免较长突发下 float32 采样序号损失小数精度。
        positions = start + torch.arange(count, device=waveform.device, dtype=torch.float64) * step
        base = torch.floor(positions).to(torch.int64)
        u = (positions - base).to(waveform.real.dtype)

        # 四点三次 Lagrange：实、虚部分别应用同样的权重。
        w_m1 = -u * (u - 1) * (u - 2) / 6
        w_0 = (u + 1) * (u - 1) * (u - 2) / 2
        w_1 = -(u + 1) * u * (u - 2) / 2
        w_2 = (u + 1) * u * (u - 1) / 6
        return (w_m1 * waveform[base - 1] + w_0 * waveform[base]
                + w_1 * waveform[base + 1] + w_2 * waveform[base + 2])
