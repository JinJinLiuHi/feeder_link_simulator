"""AWGN link simulation owned by this platform, with cancellable batch counting."""

from pathlib import Path
from decimal import Decimal
import math
import re
import time
import traceback
import numpy as np
from .catalog import custom_points
from .matrices import read_matrix, ldpc_codec


def make_modem(mod, device="cpu"):
    import torch
    from sionna.phy.mapping import Constellation, Mapper, Demapper

    options = dict(precision="single", device=device)
    if mod["kind"] in ("psk", "apsk"):
        constellation = Constellation(
            "custom",
            mod["bits"],
            points=torch.tensor(custom_points(mod), dtype=torch.complex64, device=device),
            normalize=False,
            **options,
        )
    else:
        constellation = Constellation(mod["kind"], mod["bits"], **options)
    return (
        Mapper(constellation=constellation, **options),
        Demapper("app", constellation=constellation, **options),
        constellation.points,
    )


def _polynomials(code):
    from functools import reduce

    length = code["length"]
    polys = code["polys"]
    if not 3 <= length <= (6 if code["family"] == "turbo" else 8) or len(polys) not in (2, 3):
        raise ValueError("约束长度或多项式数量无效")
    if any(not re.fullmatch(r"[0-7]{1,3}", p) for p in polys):
        raise ValueError("多项式应为八进制整数")
    values = [int(p, 8) for p in polys]
    if not all(0 < v < 2**length for v in values) or max(v.bit_length() for v in values) != length:
        raise ValueError("多项式最高次数必须与约束长度一致")

    def gcd(a, b):
        while b:
            while a and a.bit_length() >= b.bit_length():
                a ^= b << (a.bit_length() - b.bit_length())
            a, b = b, a
        return a

    if reduce(gcd, values) != 1:
        raise ValueError("多项式须在 GF(2) 上互素")
    if code["family"] == "turbo" and (len(values) != 2 or any(v % 2 == 0 for v in values)):
        raise ValueError("Turbo 反馈/前馈多项式均须有常数项")
    return tuple(format(v, f"0{length}b")[::-1] for v in values)


def make_codec(code, device="cpu", iterations=20):
    import torch

    options = dict(precision="single", device=device)
    family = code["family"]
    k = code.get("k", 1024)
    if family == "none":
        return None, None, k, k, None
    if family == "nr":
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

        encoder = LDPC5GEncoder(k, code["n"], **options)
        return (
            encoder,
            LDPC5GDecoder(
                encoder, hard_out=True, return_infobits=True, num_iter=iterations, **options
            ),
            k,
            code["n"],
            None,
        )
    if family in ("dvb", "qc", "h"):
        text = code.get("text")
        if text is None:
            text = Path(code["path"]).read_text(encoding="utf-8-sig")
        kind = "alist" if family == "dvb" else code.get("format", family)
        z = code.get("z", 1)
        if family == "qc" and code.get("path"):
            z = int(
                next(
                    line
                    for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ).split()[1]
            )
        matrix = read_matrix(text, kind, z, code.get("direction", 1))
        return ldpc_codec(matrix, device, iterations, staircase=family == "dvb")
    minimum = 1 if family == "conv" else 2
    if type(k) is not int or not minimum <= k <= 8192:
        raise ValueError(f"信息长度 K 应为 {minimum}～8192")
    polys = _polynomials(code)
    if family == "conv":
        from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder

        encoder = ConvEncoder(gen_poly=polys, rsc=False, terminate=code["terminate"], **options)
        decoder = ViterbiDecoder(encoder=encoder, method="soft_llr", **options)
    elif family == "turbo":
        from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder

        if code["rate"] not in ("1/2", "1/3") or code["interleaver"] not in (
            "qpp",
            "random",
            "custom",
        ):
            raise ValueError("Turbo 码率或交织类型无效")
        encoder = TurboEncoder(
            gen_poly=polys,
            rate=float(1 / int(code["rate"][-1])),
            terminate=code["terminate"],
            interleaver_type="3GPP" if code["interleaver"] == "qpp" else "random",
            **options,
        )
        if code["interleaver"] == "custom":
            p = code["permutation"]
            if len(p) != k or any(type(i) is not int for i in p) or sorted(p) != list(range(k)):
                raise ValueError("自定义交织须为 0～K−1 的完整排列")
            from sionna.phy import Block

            class Permutation(Block):
                def __init__(self):
                    super().__init__(**options)
                    self.register_buffer("perm_seq", torch.tensor(p, device=device))
                    self.register_buffer("perm_seq_inv", torch.argsort(self.perm_seq))
                    self.keep_state = True
                    self.keep_batch_constant = True
                    self.axis = -1

                def call(self, x, inverse=False):
                    return x.index_select(-1, self.perm_seq_inv if inverse else self.perm_seq)

            if not hasattr(encoder, "_internal_interleaver") or getattr(encoder, "built", False):
                raise ValueError("此 Sionna 版本不支持当前自定义交织适配")
            fixed = Permutation()
            encoder._internal_interleaver = fixed
        decoder = TurboDecoder(
            encoder=encoder, num_iter=iterations, algorithm="log", hard_out=True, **options
        )
        if (
            code["interleaver"] == "custom"
            and getattr(decoder, "_internal_interleaver", None) is not fixed
        ):
            raise ValueError("Turbo 编译码交织器不一致")
    else:
        raise ValueError("未知编码类型")
    with torch.no_grad():
        n = encoder(torch.zeros((1, k), device=device)).shape[-1]
    return encoder, decoder, k, n, None


def validate_code(code):
    import torch

    torch.set_num_threads(4)
    encoder, decoder, k, n, h = make_codec(code, iterations=6)
    if k <= 0 or n <= k:
        raise ValueError("编码必须含信息位和校验位")
    samples = torch.zeros((4, k))
    samples[1] = 1
    samples[2, 0] = 1
    samples[3] = torch.randint(0, 2, (k,))
    with torch.no_grad():
        for u in samples.split(1):
            word = encoder(u)
            if word.shape != (1, n) or not torch.all((word == 0) | (word == 1)):
                raise ValueError("编码输出长度或比特值异常")
            if h is not None and np.any(np.asarray(h @ word.cpu().numpy().T, dtype=np.int64) % 2):
                raise ValueError("编码结果不满足原校验约束")
            decoded = decoder((word * 2 - 1) * 16)[..., :k]
            if not torch.equal(decoded, u):
                raise ValueError("全零/全一/单脉冲/随机信息无噪声恢复失败")
    return {**code, "k": int(k), "n": int(n)}


def snr_grid(request):
    start, stop, step = (Decimal(str(request[key])) for key in ("start", "stop", "step"))
    if (
        not all(v.is_finite() for v in (start, stop, step))
        or step <= 0
        or start > stop
        or start < -30
        or stop > 60
    ):
        raise ValueError("Eb/N0 范围应在 −30～60 dB 内，步进为正，终点不小于起点")
    count = int((stop - start) // step) + 1
    if count > 1001:
        raise ValueError("单次最多 1001 个信噪比点")
    return [float(start + i * step) for i in range(count)]


def simulate(request, emit, cancelled):
    import torch
    from sionna.phy import config
    from sionna.phy.channel import AWGN

    grid = snr_grid(request)
    budget = request["budget"]
    target = request["errors"]
    if (
        type(budget) is not int
        or not 1 <= budget <= 10**12
        or type(target) is not int
        or target < 0
    ):
        raise ValueError("比特预算或错误目标无效")
    if cancelled():
        emit({"type": "done", "cancelled": True})
        return
    config.seed = None
    code = request["code"]
    mod = request["mod"]
    requested = request.get("device", "auto")
    device = (
        ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    )
    if code["family"] in ("conv", "turbo") and requested == "auto":
        device = "cpu"
    torch.set_num_threads(4)
    emit({"type": "status", "message": "正在准备调制与编译码…"})
    encoder, decoder, k, n, h = make_codec(code, device, request["iterations"])
    mapper, demapper, _ = make_modem(mod, device)
    symbols = math.ceil(n / mod["bits"])
    padding = symbols * mod["bits"] - n
    batch = min(
        request.get("batch", 0) or 16,
        32,
        max(1, budget // k),
        max(1, 1000000 // (symbols * (1 << mod["bits"]))),
    )
    if code["family"] in ("conv", "turbo"):
        batch = min(
            batch, max(1, 500000 // (k * 2 ** (code["length"] - 1) * request["iterations"]))
        )
    if h is not None:
        padded = h.shape[0] * int(np.diff(h.indptr).max()) + h.shape[1] * int(
            np.asarray(h.sum(axis=0)).max()
        )
        if padded > 4000000:
            raise ValueError("译码节点规模超过内存限制")
        batch = min(batch, max(1, 2000000 // max(1, padded)))
    awgn = AWGN(precision="single", device=device)

    class Cancelled(Exception):
        pass

    def check(*args):
        if cancelled():
            raise Cancelled()

    hooks = []
    # Include the internal BCJR/trellis modules so Stop can interrupt between iterations.
    for block in (encoder, decoder, mapper, demapper, awgn):
        if hasattr(block, "modules"):
            hooks.extend(m.register_forward_pre_hook(check) for m in block.modules())
    try:
        with torch.no_grad():
            for index, ebno in enumerate(grid):
                if cancelled():
                    break
                counts = dict(bits=0, frames=0, bit_errors=0, frame_errors=0)
                began = time.perf_counter()
                reason = "比特预算"
                noise = torch.tensor(
                    symbols / k / 10 ** (ebno / 10), dtype=torch.float32, device=device
                )
                while counts["bits"] < budget:
                    if cancelled():
                        reason = "已停止"
                        break
                    valid = min(budget - counts["bits"], batch * k)
                    frames = math.ceil(valid / k)
                    u = torch.randint(0, 2, (frames, k), device=device).float()
                    u.reshape(-1)[valid:] = 0
                    try:
                        word = u if encoder is None else encoder(u)
                        word = torch.nn.functional.pad(word, (0, padding)) if padding else word
                        rx = awgn(mapper(word), noise)
                        llr = demapper(rx, noise)[..., :n]
                        decoded = llr > 0 if decoder is None else decoder(llr)[..., :k]
                    except Cancelled:
                        reason = "已停止"
                        break
                    errors = decoded.bool() != u.bool()
                    errors.reshape(-1)[valid:] = False
                    counts["bits"] += valid
                    counts["frames"] += frames
                    counts["bit_errors"] += int(errors.sum())
                    counts["frame_errors"] += int(errors.any(dim=1).sum())
                    emit(
                        {
                            "type": "progress",
                            "index": index,
                            "points": len(grid),
                            "ebno": ebno,
                            **counts,
                        }
                    )
                    if target and counts["bit_errors"] >= target:
                        reason = "错误目标"
                        break
                if counts["bits"]:
                    ber = counts["bit_errors"] / counts["bits"]
                    fer = counts["frame_errors"] / counts["frames"]
                    upper = (
                        -math.expm1(math.log(0.05) / counts["bits"])
                        if counts["bit_errors"] == 0
                        else None
                    )
                    emit(
                        {
                            "type": "point",
                            "row": dict(
                                ebno=ebno,
                                ber=ber,
                                fer=fer,
                                upper=upper,
                                seconds=time.perf_counter() - began,
                                reason=reason,
                                **counts,
                            ),
                        }
                    )
                if cancelled():
                    break
    finally:
        for hook in hooks:
            hook.remove()
    emit({"type": "done", "cancelled": cancelled()})


def worker(action, request, queue, cancel):
    try:
        if cancel.is_set():
            queue.put({"type": "done", "cancelled": True})
            return
        queue.put({"type": "status", "message": "正在加载仿真计算库…"})
        if action == "validate":
            queue.put({"type": "validated", "code": validate_code(request)})
        else:
            simulate(request, queue.put, cancel.is_set)
    except Exception as error:
        queue.put({"type": "error", "message": str(error), "detail": traceback.format_exc()})
