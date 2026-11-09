"""Lightweight frame plans shared by the source editor and waveform runtime."""

import copy
from functools import lru_cache
import json
from pathlib import Path
import re

from link_budget.algorithm.ber_core.catalog import session_catalog

DEFAULT_HEADER_HEX = "1ACFFC1D"


def legacy_frame(settings):
    return not any(key in settings for key in ("business_frame", "payload_symbols", "header_hex")) and any(
        key in settings for key in ("business_frames", "words", "k", "n", "modulation"))


def header_hex(settings):
    value = settings.get("header_hex", DEFAULT_HEADER_HEX)
    if not isinstance(value, str):
        raise ValueError("帧头须为十六进制文本")
    value = re.sub(r"\s+", "", value).removeprefix("0x").removeprefix("0X").upper()
    if not re.fullmatch(r"[0-9A-F]{1,256}", value):
        raise ValueError("帧头须为 1～256 位十六进制字符，例如 1ACFFC1D")
    return value


def header_bits(settings):
    """MSB first, four BPSK symbols per hex digit, retaining leading zeros."""
    return [int(bit) for digit in header_hex(settings) for bit in f"{int(digit, 16):04b}"]


def payload_size(settings):
    from .waveform_chain import _number, MAX_SAMPLES
    return None if legacy_frame(settings) else _number(settings, "payload_symbols", 1024, 1, MAX_SAMPLES, True)


def editor_settings(settings):
    """Editing an older mixed plan explicitly adopts the shared single-row form."""
    result = copy.deepcopy(settings)
    rows = result.pop("business_frames", None)
    result.pop("words", None)
    result.setdefault("business_frame", copy.deepcopy(rows[0][0]) if rows else default_business_frame())
    result.setdefault("payload_symbols", 1024)
    result.setdefault("header_hex", DEFAULT_HEADER_HEX)
    return result


def default_business_frame():
    return dict(mod=dict(name="QPSK", kind="qam", bits=2),
                code=dict(id="nr_512_1024", name="K512 · N1024 · 1/2", family="nr", k=512, n=1024))


def symbol_rate(settings):
    from .waveform_chain import _number
    if "symbol_rate" in settings:
        return _number(settings, "symbol_rate", 700e6, 1., 1e12)
    fs = _number(settings, "fs", 2.8e9, 1., 1e12)
    sps = _number(settings, "sps", 4, 2, 256, True)
    return fs/sps


def frame_plan(settings):
    from .waveform_chain import _number
    segments = _number(settings, "segments", 4, 1, 32, True)
    if not legacy_frame(settings):
        item = settings.get("business_frame", default_business_frame())
        if not isinstance(item, dict) or not isinstance(item.get("mod"), dict) or not isinstance(item.get("code"), dict):
            raise ValueError("业务帧须选择调制和编码")
        return [[copy.deepcopy(item)] for _ in range(segments)]
    words = _number(settings, "words", 2, 1, 32, True)
    rows = settings.get("business_frames")
    if rows is None:
        item = default_business_frame()
        if "k" in settings or "n" in settings:
            item["code"].update(k=settings.get("k", 512), n=settings.get("n", 1024))
            item["code"]["id"] = f"nr_{item['code']['k']}_{item['code']['n']}"
            item["code"]["name"] = f"K{item['code']['k']} · N{item['code']['n']}"
        name = settings.get("modulation", "QPSK")
        mods = {m["name"]: m for m in session_catalog()["modulations"]}
        item["mod"] = mods.get(name, dict(name=name, legacy_name=name))
        rows = [[copy.deepcopy(item) for _ in range(words)] for _ in range(segments)]
    if not isinstance(rows, list) or len(rows) != segments:
        raise ValueError("业务帧配置与每突发帧数不一致")
    for group in rows:
        if not isinstance(group, list) or len(group) != words:
            raise ValueError("业务帧配置与每帧业务帧数不一致")
        for item in group:
            if not isinstance(item, dict) or not isinstance(item.get("mod"), dict) or not isinstance(item.get("code"), dict):
                raise ValueError("每个业务帧均须选择调制和编码")
    return copy.deepcopy(rows)


def modulation_bits(mod):
    if "legacy_name" in mod:
        from .burst_link.modulation import create
        return create(mod["legacy_name"]).bits_per_symbol
    bits = mod.get("bits")
    if type(bits) is not int or not 1 <= bits <= 10:
        raise ValueError("调制每符号位数必须为 1～10")
    return bits


@lru_cache(maxsize=128)
def _matrix_dimensions(definition):
    from link_budget.algorithm.ber_core.matrices import read_matrix, systematic_form
    code = json.loads(definition)
    text = code.get("text")
    if text is None:
        text = Path(code["path"]).read_text(encoding="utf-8-sig")
    z = code.get("z", 1)
    if code["family"] == "qc" and code.get("path"):
        z = int(next(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")).split()[1])
    matrix = read_matrix(text, code.get("format", code["family"]), z, code.get("direction", 1))
    _, relation, _ = systematic_form(matrix)
    return relation.shape[1], matrix.shape[1]


def code_dimensions(code):
    if "k" in code and "n" in code:
        k, n = code["k"], code["n"]
    else:
        k, n = _matrix_dimensions(json.dumps(code, sort_keys=True))
    if type(k) is not int or type(n) is not int or not 0 < k <= n <= 100000:
        raise ValueError("编码信息长度 K、输出长度 N 无效")
    return k, n


def portable_code(code):
    code = copy.deepcopy(code)
    if code.get("path"):
        path = Path(code["path"])
        if "text" not in code:
            code["text"] = path.read_text(encoding="utf-8-sig")
        if code["family"] == "qc":
            code["z"] = int(next(line for line in code["text"].splitlines()
                                 if line.strip() and not line.lstrip().startswith("#")).split()[1])
        code["k"], code["n"] = code_dimensions(code)
        code.pop("path")
    return code


def frame_summary(settings):
    rows = frame_plan(settings)
    field_size = payload_size(settings)
    header_count = 64 if legacy_frame(settings) else len(header_bits(settings))
    information = coded = padding = data_symbols = 0
    frames = []
    for i, group in enumerate(rows):
        for j, item in enumerate(group):
            k, n = code_dimensions(item["code"])
            m = modulation_bits(item["mod"])
            length = (n + m - 1) // m
            information += k
            coded += n
            padding += length*m - n
            data_symbols += length
            frames.append(dict(segment=i+1, frame=j+1, information_bits=k, coded_bits=n,
                               symbols=length, padding_bits=length*m-n,
                               modulation=item["mod"]["name"], coding=item["code"]["name"]))
    if field_size is not None and any(f["symbols"] > field_size for f in frames):
        largest = max(f["symbols"] for f in frames)
        raise ValueError(f"业务帧编码调制后需要 {largest:,} 个符号，超过设置的业务数据符号数 {field_size:,}")
    field_symbols = data_symbols if field_size is None else field_size*len(rows)
    header = 1024 + header_count*len(rows)
    from .waveform_chain import MAX_SAMPLES
    if header+field_symbols > MAX_SAMPLES:
        raise ValueError("完整突发超过最大符号数，请减少每突发帧数或业务数据符号数")
    return dict(information_bits=information, coded_bits=coded, modulation_padding_bits=padding,
                known_header_bits=header, header_symbols=header_count,
                coded_symbols=data_symbols, payload_symbols=field_symbols,
                zero_padding_symbols=field_symbols-data_symbols, total_symbols=header+field_symbols,
                symbol_rate=symbol_rate(settings), frames=frames)
