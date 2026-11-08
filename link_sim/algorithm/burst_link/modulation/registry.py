"""按名称选择已登记的调制方案。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ._mapping import ConstellationModulator
from .bpsk import bpsk
from .qpsk import qpsk
from .psk8 import psk8
from .qam16 import qam16
from .qam64 import qam64
from .qam256 import qam256
from .qam1024 import qam1024
from .apsk16 import apsk16
from .apsk32 import apsk32
from .apsk64 import apsk64
from .apsk256 import apsk256


# 每项直接指向具体调制函数；名称、元信息和函数共同构成一条登记。
# create 每次调用该函数，返回独立调制器。
_REGISTRY: Mapping[str, tuple[Callable[[], ConstellationModulator], str, int, str | None]] = MappingProxyType({
    "BPSK": (bpsk, "PSK", 2, None),
    "QPSK": (qpsk, "PSK", 4, None),
    "8PSK": (psk8, "PSK", 8, None),
    "16QAM": (qam16, "QAM", 16, None),
    "64QAM": (qam64, "QAM", 64, None),
    "256QAM": (qam256, "QAM", 256, None),
    "1024QAM": (qam1024, "QAM", 1024, None),
    "16APSK": (apsk16, "APSK", 16, "3/4"),
    "32APSK": (apsk32, "APSK", 32, "3/4"),
    "64APSK": (apsk64, "APSK", 64, "128/180"),
    "256APSK": (apsk256, "APSK", 256, "116/180"),
})


def available() -> tuple[str, ...]:
    """返回已登记的调制方案名称。"""
    return tuple(_REGISTRY)


def _lookup(name: str) -> tuple[Callable[[], ConstellationModulator], str, int, str | None]:
    if not isinstance(name, str):
        raise TypeError("调制方案名称必须是字符串")
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"未知调制方案 {name!r}；可通过 modulation.available() 查看可用名称") from None


def describe(name: str) -> dict[str, object]:
    """返回方案参数；APSK 的 profile_rate 仅标识星座预设。"""
    _, family, order, profile_rate = _lookup(name)
    return {
        "name": name,
        "family": family,
        "order": order,
        "bits_per_symbol": order.bit_length() - 1,
        "profile_rate": profile_rate,
    }


def create(name: str) -> ConstellationModulator:
    """根据登记名称创建调制器。"""
    factory, _, _, _ = _lookup(name)
    return factory()
