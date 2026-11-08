"""Local modulation/code definitions; no desktop-reference imports."""

from pathlib import Path
from fractions import Fraction
import cmath
import math
import re
import json
import copy

MATRIX_ROOT = Path(__file__).resolve().parents[1] / "data" / "ldpc"
FAMILIES = {
    "dvb": "LDPC · DVB-S2",
    "nr": "LDPC · 5G NR",
    "qc": "LDPC · QC 基矩阵",
    "h": "LDPC · H / alist",
    "conv": "卷积码",
    "turbo": "Turbo",
    "none": "不编码",
}


def modulations():
    return [
        {"name": name, "kind": kind, "bits": bits}
        for name, kind, bits in [
            ("BPSK", "pam", 1),
            ("QPSK", "qam", 2),
            ("8PSK", "psk", 3),
            ("16QAM", "qam", 4),
            ("64QAM", "qam", 6),
            ("256QAM", "qam", 8),
            ("1024QAM", "qam", 10),
        ]
    ]


def apsk_definition(order, counts, radii, phases):
    if order not in (4, 8, 16, 32, 64, 128, 256):
        raise ValueError("APSK 阶数应为 4、8、16、32、64、128 或 256")
    if not counts or len(counts) != len(radii) or len(counts) != len(phases):
        raise ValueError("每一环都需要点数、半径和初始相位")
    if any(type(n) is not int or n <= 0 for n in counts) or sum(counts) != order:
        raise ValueError("各环点数必须是正整数，且总和等于星座阶数")
    if not all(math.isfinite(v) for v in radii + phases):
        raise ValueError("半径和相位必须是有限数值")
    if radii[0] <= 0 or any(a >= b for a, b in zip(radii, radii[1:])):
        raise ValueError("半径必须为正数，且由内向外递增")
    return {
        "name": f"{order}APSK · {'+'.join(map(str, counts))} · r{'/'.join(f'{r:g}' for r in radii)} · φ{'/'.join(f'{p:g}' for p in phases)}",
        "kind": "apsk",
        "bits": order.bit_length() - 1,
        "counts": counts,
        "radii": radii,
        "phases": phases,
    }


def custom_points(mod):
    size = 1 << mod["bits"]
    points = [0j] * size
    if mod["kind"] == "psk":
        geometry = [cmath.exp(2j * math.pi * i / size) for i in range(size)]
    else:
        checked = apsk_definition(size, mod["counts"], mod["radii"], mod["phases"])
        geometry = [
            r * cmath.exp(1j * (math.radians(p) + 2 * math.pi * i / n))
            for n, r, p in zip(checked["counts"], checked["radii"], checked["phases"])
            for i in range(n)
        ]
        if len({(round(p.real, 10), round(p.imag, 10)) for p in geometry}) != size:
            raise ValueError("星座包含重合点")
    scale = math.sqrt(sum(abs(p) ** 2 for p in geometry) / size)
    for i, point in enumerate(geometry):
        points[i ^ (i >> 1)] = point / scale
    return points


def load_catalog():
    """Read lightweight local definitions without initializing the compute runtime."""
    codes = []
    for path in sorted((MATRIX_ROOT / "DVB-S2").glob("*/*.alist")):
        frame = path.parent.name
        token = re.search(r"_r([0-9]+)", path.stem).group(1)
        rates = [
            Fraction(int(token[:i]), int(token[i:]))
            for i in range(1, len(token))
            if 0 < int(token[:i]) < int(token[i:]) <= 10
        ]
        if len(rates) != 1:
            raise ValueError(f"矩阵名称码率不明确：{path.name}")
        with path.open(encoding="utf-8") as stream:
            n, m = map(int, stream.readline().split())
        rate = str(rates[0])
        frame_label = "短帧" if frame == "short" else "常规帧"
        codes.append(
            dict(
                id=path.stem,
                name=f"{frame_label} · {rate}",
                family="dvb",
                k=n - m,
                n=n,
                rate=rate,
                path=str(path),
            )
        )
    profile_file = MATRIX_ROOT.parent / "nr_profiles.json"
    profiles = json.loads(profile_file.read_text(encoding="utf-8"))["profiles"]
    codes.extend(profiles)
    if not any(c.get("k") == 512 and c.get("n") == 1024 for c in codes):
        codes.append(dict(id="nr_512_1024", name="K512 · N1024 · 1/2", family="nr", k=512, n=1024))
    for path in sorted(MATRIX_ROOT.glob("**/*.base")):
        codes.append(
            dict(
                id=path.stem,
                name=path.stem.replace("qc_ldpc_", "").upper(),
                family="qc",
                path=str(path),
            )
        )
    codes.append(dict(id="none", name="不编码", family="none", k=1024, n=1024))
    return dict(
        modulations=modulations(),
        codes=codes,
        environment="计算设备在开始仿真时检测",
    )


_session_catalog = None


def session_catalog():
    """Share the BER definitions with waveform editors during this app session."""
    global _session_catalog
    if _session_catalog is None:
        _session_catalog = load_catalog()
    return copy.deepcopy(_session_catalog)


def update_session_catalog(modulations, codes):
    global _session_catalog
    _session_catalog = dict(modulations=copy.deepcopy(modulations), codes=copy.deepcopy(codes))
