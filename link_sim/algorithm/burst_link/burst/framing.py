"""按固定结构生成一个符号级复基带突发。"""

from __future__ import annotations

import operator
from typing import Any, Iterable

import torch
import torch.nn.functional as functional


def _integer(value: Any, name: str, *, minimum: int) -> int:
    """接受整数类型，排除布尔值和不满足范围的配置。"""
    if isinstance(value, bool):
        raise TypeError(f"{name} 必须为整数")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} 必须为整数") from error
    if result < minimum:
        raise ValueError(f"{name} 必须不小于 {minimum}")
    return result


def _binary_1d(value: Any, name: str, device: torch.device, *, nonempty: bool) -> torch.Tensor:
    """把一维二进制输入放到指定设备，避免静默截断非二进制值。"""
    try:
        bits = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} 必须是可转换为张量的一维比特数组") from error
    if bits.ndim != 1 or bits.is_complex():
        raise ValueError(f"{name} 必须是一维实数或布尔比特数组")
    if nonempty and bits.numel() == 0:
        raise ValueError(f"{name} 不能为空")
    if bits.numel() and not bool(torch.all((bits == 0) | (bits == 1)).item()):
        raise ValueError(f"{name} 只能包含 0 和 1")
    return bits.to(dtype=torch.float32)


def _header_symbols(value: Any, name: str, device: torch.device) -> torch.Tensor:
    """验证并固定调用方生成的一维复符号，不隐式迁移已有张量。"""
    if isinstance(value, torch.Tensor) and value.device != device:
        raise ValueError(f"{name}必须与组帧对象在同一设备")
    try:
        symbols = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name}必须是可转换为张量的一维复数数组") from error
    if symbols.ndim != 1 or not symbols.is_complex():
        raise ValueError(f"{name}必须是一维复数数组")
    if symbols.numel() == 0:
        raise ValueError(f"{name}不能为空")
    if not bool(torch.isfinite(symbols).all().item()):
        raise ValueError(f"{name}不能包含非有限值")
    return symbols.detach().clone()


def _symbols(modulator: Any, bits: torch.Tensor, expected: tuple[int, ...], name: str) -> torch.Tensor:
    result = modulator.modulate(bits)
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != expected:
        raise ValueError(f"{name} 输出尺寸不匹配，预期 {expected}")
    if result.device != bits.device or not result.is_complex():
        raise ValueError(f"{name} 必须返回与输入同设备的复数张量")
    return result


class BurstFramer:
    """组装一个固定结构的符号级突发；不执行过采样或成型滤波。

    同步头和帧头由调用方提供复符号，编码器处理 [W, K] 信息块。
    分块/补零属于组帧，编码输出的调制补位按每码字独立处理。
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_configuration_locked", False):
            raise AttributeError("组帧配置已确定；修改配置请重新实例化")
        object.__setattr__(self, name, value)

    def __init__(self, *, sync_symbols: Any, header_symbols: Any,
                 codewords_per_frame: Iterable[int], data_modulation: Any,
                 encoder: Any, device: str | torch.device | None = None) -> None:
        self.encoder = encoder
        self.data_modulation = data_modulation

        selected_device = device if device is not None else getattr(encoder, "device", "cpu")
        self.device = torch.device(selected_device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        encoder_device = getattr(encoder, "device", None)
        if encoder_device is not None:
            encoder_device = torch.device(encoder_device)
            if encoder_device.type == "cuda" and encoder_device.index is None:
                encoder_device = torch.device("cuda", torch.cuda.current_device())
            if encoder_device != self.device:
                raise ValueError("组帧设备必须与编码器设备一致")

        if not callable(getattr(data_modulation, "modulate", None)):
            raise TypeError("数据调制器必须提供 modulate(bits)")
        if not callable(getattr(encoder, "encode", None)):
            raise TypeError("编码器必须提供 encode(info_words)")

        try:
            counts = tuple(codewords_per_frame)
        except TypeError as error:
            raise TypeError("每段码字数必须为非空整数序列") from error
        if not counts:
            raise ValueError("至少需要一个帧段")
        self.codewords_per_frame = tuple(
            _integer(count, f"第 {i + 1} 段码字数", minimum=0)
            for i, count in enumerate(counts)
        )
        self.codeword_count = sum(self.codewords_per_frame)
        self.information_bits_per_codeword = _integer(
            getattr(encoder, "information_bits_per_codeword", None), "编码器信息块长度", minimum=1)
        self.encoded_bits_per_codeword = _integer(
            getattr(encoder, "encoded_bits_per_codeword", None), "编码器输出码字长度", minimum=1)
        self.bits_per_symbol = _integer(
            getattr(data_modulation, "bits_per_symbol", None), "数据调制每符号位数", minimum=1)
        self._sync_symbols = _header_symbols(sync_symbols, "同步头符号", self.device)
        self._header_symbols = _header_symbols(header_symbols, "帧头符号", self.device)
        if self._sync_symbols.dtype != self._header_symbols.dtype:
            raise ValueError(f"同步头与帧头符号的复数精度必须一致，当前为 {self._sync_symbols.dtype} 和 {self._header_symbols.dtype}")

        k = self.information_bits_per_codeword
        n = self.encoded_bits_per_codeword
        m = self.bits_per_symbol
        self.symbols_per_codeword = (n + m - 1) // m
        self.modulation_padding_bits = self.symbols_per_codeword * m - n
        self.capacity_bits = self.codeword_count * k
        self.frame_capacities = tuple(count * k for count in self.codewords_per_frame)
        self.sync_symbol_count = self._sync_symbols.numel()
        self.header_symbol_count = self._header_symbols.numel()
        self.frame_symbol_lengths = tuple(
            self.header_symbol_count + count * self.symbols_per_codeword
            for count in self.codewords_per_frame
        )
        self.total_symbols = self.sync_symbol_count + sum(self.frame_symbol_lengths)

        cursor = self.sync_symbol_count
        headers: list[tuple[int, int]] = []
        frames: list[tuple[int, int]] = []
        words: list[tuple[tuple[int, int], ...]] = []
        for count in self.codewords_per_frame:
            frame_start = cursor
            headers.append((cursor, cursor + self.header_symbol_count))
            cursor += self.header_symbol_count
            row: list[tuple[int, int]] = []
            for _ in range(count):
                row.append((cursor, cursor + self.symbols_per_codeword))
                cursor += self.symbols_per_codeword
            words.append(tuple(row))
            frames.append((frame_start, cursor))
        self.header_spans = tuple(headers)
        self.codeword_spans = tuple(words)
        self.frame_spans = tuple(frames)
        self._configuration_locked = True

    @property
    def header_symbols(self) -> torch.Tensor:
        """返回训练帧头的独立副本，用于接收处理。"""
        return self._header_symbols.clone()

    @property
    def sync_symbols(self) -> torch.Tensor:
        """返回同步符号的独立副本，用于接收同步。"""
        return self._sync_symbols.clone()

    @torch.no_grad()
    def __call__(self, info_bits: Any) -> torch.Tensor:
        """组帧一组信息比特，返回指定设备上一维复符号张量。"""
        bits = _binary_1d(info_bits, "信息比特", self.device, nonempty=False)
        if bits.numel() > self.capacity_bits:
            raise ValueError(f"信息位数 {bits.numel()} 超过单突发容量 {self.capacity_bits}")
        if (_integer(getattr(self.encoder, "information_bits_per_codeword", None), "编码器信息块长度", minimum=1)
                != self.information_bits_per_codeword
                or _integer(getattr(self.encoder, "encoded_bits_per_codeword", None), "编码器输出码字长度", minimum=1)
                != self.encoded_bits_per_codeword
                or _integer(getattr(self.data_modulation, "bits_per_symbol", None), "数据调制每符号位数", minimum=1)
                != self.bits_per_symbol):
            raise RuntimeError("编码器或调制器尺寸已改变，请重新实例化组帧对象")

        k, n = self.information_bits_per_codeword, self.encoded_bits_per_codeword
        if self.codeword_count:
            words = torch.zeros((self.codeword_count, k), dtype=torch.float32, device=self.device)
            words.reshape(-1)[:bits.numel()] = bits
            coded = self.encoder.encode(words)
            if not isinstance(coded, torch.Tensor) or tuple(coded.shape) != (self.codeword_count, n):
                raise ValueError(f"编码器输出尺寸必须为 {(self.codeword_count, n)}")
            if coded.device != self.device or coded.is_complex():
                raise ValueError("编码器必须返回与组帧对象同设备的实数或布尔比特张量")
            if self.modulation_padding_bits:
                coded = functional.pad(coded, (0, self.modulation_padding_bits), value=0)
            payload = _symbols(self.data_modulation, coded,
                               (self.codeword_count, self.symbols_per_codeword), "数据调制器")
            if payload.dtype != self._sync_symbols.dtype:
                raise ValueError(f"数据调制器输出与头部符号的复数精度必须一致，当前为 {payload.dtype} 和 {self._sync_symbols.dtype}")
        else:
            payload = torch.empty((0, self.symbols_per_codeword), dtype=self._sync_symbols.dtype, device=self.device)

        result = torch.empty(self.total_symbols, dtype=self._sync_symbols.dtype, device=self.device)
        result[:self.sync_symbol_count] = self._sync_symbols
        row = 0
        for count, header_span, frame_span in zip(self.codewords_per_frame, self.header_spans, self.frame_spans):
            result[header_span[0]:header_span[1]] = self._header_symbols
            if count:
                result[header_span[1]:frame_span[1]] = payload[row:row + count].reshape(-1)
            row += count
        return result
