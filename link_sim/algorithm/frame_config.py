"""Lightweight frame plans shared by the source editor and waveform runtime."""

import copy
from functools import lru_cache
import json
from pathlib import Path

from link_budget.algorithm.ber_core.catalog import session_catalog


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
    header = 1024 + 64*len(rows)
    return dict(information_bits=information, coded_bits=coded, modulation_padding_bits=padding,
                known_header_bits=header, total_symbols=header+data_symbols,
                symbol_rate=symbol_rate(settings), frames=frames)
