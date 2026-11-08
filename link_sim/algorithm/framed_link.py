"""Adapters for the reference project's framed, complex-envelope receiver."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import max_len_seq


# Keep the established source scale for the downstream fitted device models.
FRAME_SYMBOL_SCALE = .2


@dataclass
class BurstContext:
    framer: object
    shaper: object
    bits: np.ndarray
    symbols: np.ndarray
    lane: int


def frame_source(settings):
    import torch
    from .burst_link.modulation import create
    from .frame_config import frame_plan, frame_summary
    from .mixed_burst import MixedBurstFramer
    from .waveform_chain import Wave, _number

    # Small burst kernels avoid CPU thread oversubscription in the GUI worker.
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    lane = _number(settings, "lane", 0, 0, 1, True)
    seed = _number(settings, "seed", 2026 + lane, 0, 2**32-1, True)
    sync = max_len_seq(7, taps=[6] if lane == 0 else [4], state=[1]*7)[0]
    header = max_len_seq(6, taps=[5] if lane == 0 else [1], state=[1]*6)[0]
    bpsk = create("BPSK")
    sync_symbols = bpsk.modulate(torch.tensor(np.tile(np.r_[sync, 0], 8), dtype=torch.uint8))
    header_symbols = bpsk.modulate(torch.tensor(np.r_[header, 0], dtype=torch.uint8))
    from .waveform_chain import MAX_SAMPLES
    summary = frame_summary(settings)
    if summary["total_symbols"] > MAX_SAMPLES:
        raise ValueError("完整突发超过最大符号数，请减少每突发帧数或每帧业务帧数")
    plan = frame_plan(settings)
    framer = MixedBurstFramer(sync_symbols, header_symbols, plan)
    bits = np.random.default_rng(seed).integers(0, 2, framer.capacity_bits, dtype=np.uint8)
    symbols = framer(torch.tensor(bits)).numpy()
    context = BurstContext(framer, None, bits, symbols, lane)
    return Wave(FRAME_SYMBOL_SCALE*symbols, summary["symbol_rate"], context=context, metrics=summary,
                notes=(f"同步头 1024 符号；每突发 {len(plan)} 帧，每帧 64 符号帧头＋{len(plan[0])} 个业务帧；"
                       f"原始业务 {len(bits):,} bit，编码后 {summary['coded_bits']:,} bit，"
                       f"总计 {len(symbols):,} 符号；码元速率 {summary['symbol_rate']/1e6:g} Mbaud",))


def tensor(x):
    import torch
    return torch.as_tensor(np.asarray(x), dtype=torch.complex128)


def synchronize(wave, threshold):
    from .burst_link.synchronization import CorrelationSynchronizer
    if wave.context is None:
        raise ValueError("同步需要带有约定帧结构的信号源")
    c = wave.context
    if c.shaper is None:
        raise ValueError("同步前请连接并启用独立的发射 RRC 模块")
    sync = CorrelationSynchronizer(sync_symbols=c.framer.sync_symbols, shaper=c.shaper, threshold=threshold)
    result = sync(tensor(wave.x))
    if not result.success:
        raise ValueError(f"同步失败：{result.reason}；相关峰 {result.peak_value}")
    start = result.fractional_start_sample if result.fractional_valid else float(result.start_sample)
    return start, result


def process_framed_stage(stage, wave):
    from .waveform_chain import Wave, _number
    from .burst_link.receive_filtering import RRCMatchedFilter
    from .burst_link.interpolation import FractionalInterpolator
    from .burst_link.channel.geo_wideband import GEOWidebandChannel

    c, p = wave.context, stage.settings
    if stage.kind in ("rrc_tx", "rrc_rx", "sync", "sink") and c is None:
        raise ValueError("请连接按帧生成的发射信号")
    if stage.kind == "rrc_tx":
        from .rrc_stage import transmit_shaper
        from .waveform_chain import MAX_SAMPLES
        shaper = transmit_shaper(stage)
        if shaper.output_length(len(wave.x)) > MAX_SAMPLES:
            raise ValueError("RRC 成形后超过最大采样点数，请减少业务帧数或每符号采样点")
        x = shaper(tensor(wave.x)).numpy()
        return Wave(x, wave.fs*shaper.samples_per_symbol, wave.carrier_hz,
                    context=replace(c, shaper=shaper), metrics=wave.metrics,
                    notes=(f"独立 RRC 时域成形：sps={shaper.samples_per_symbol}，α={shaper.rolloff:g}，"
                           f"span={shaper.span_symbols}；保留完整卷积尾部",))
    if stage.kind == "rrc_rx":
        if c.shaper is None:
            raise ValueError("接收 RRC 需要上游已启用的发射 RRC")
        if stage.model is not None:
            from .rrc_stage import SavedRRC
            rx = SavedRRC(stage.model)
            if (rx.samples_per_symbol != c.shaper.samples_per_symbol or rx.tap_count != c.shaper.tap_count
                    or not np.allclose(rx.coefficients().numpy(), c.shaper.coefficients().numpy())):
                raise ValueError("接收 RRC 模型应与发射 RRC 的抽头和每符号采样点一致")
        x = RRCMatchedFilter(c.shaper)(tensor(wave.x)).numpy()
        return Wave(x, wave.fs, wave.carrier_hz, context=c, notes=("接收 RRC 匹配滤波，保留完整卷积尾部",))
    if stage.kind == "sync":
        start, result = synchronize(wave, _number(p, "threshold", .4, .01, 1.))
        x = FractionalInterpolator()(tensor(wave.x), start=start, step=c.shaper.samples_per_symbol,
                                    count=c.framer.total_symbols).numpy()
        return Wave(x, wave.fs/c.shaper.samples_per_symbol, wave.carrier_hz, context=c,
                    metrics=dict(sync_sample=start, correlation=result.peak_value),
                    notes=(f"同步位置 {start:.5f} 采样；相关峰 {result.peak_value:.5f}；小数定时插值",))
    if stage.kind == "sink":
        bits = c.framer.decode(tensor(wave.x)).numpy()
        tracked = c.framer.last_phase_result.symbols.numpy()
        spans = [span for group in c.framer.codeword_spans for span in group]
        data = np.concatenate([tracked[a:b] for a, b in spans])
        reference = np.concatenate([c.symbols[a:b] for a, b in spans])
        errors = int(np.count_nonzero(bits != c.bits))
        evm = float(np.linalg.norm(data-reference)/np.linalg.norm(reference))
        return Wave(tracked, wave.fs, wave.carrier_hz, context=c, decoded_bits=bits,
                    metrics=dict(wave.metrics, bit_errors=errors, bit_count=len(bits), ber=errors/len(bits), evm=evm),
                    notes=(f"帧头相位/幅度恢复，各业务帧按各自调制编码软解调及译码；误码 {errors}/{len(bits)}，"
                           f"BER={errors/len(bits):.4g}；业务符号 EVM={100*evm:.3f}%",))
    if stage.kind == "channel":
        channel = GEOWidebandChannel(sample_rate_hz=wave.fs, filter_taps=[1.],
            delay_samples=_number(p, "delay", 0., 0., 4096.),
            frequency_offset_hz=_number(p, "offset_hz", 0., -wave.fs/2+1, wave.fs/2-1),
            amplitude=10**(_number(p, "gain_db", 0., -100., 100.)/20))
        x = channel(tensor(wave.x)).numpy()
        x = add_noise(x, p, 2026 + (c.lane if c is not None else 0))
        return Wave(x, wave.fs, wave.carrier_hz, context=c, notes=("无线信道含接收窗口保护、增益、频偏、短时延及复高斯噪声",))
    raise ValueError("未知收发模块")


def add_noise(x, settings, seed):
    from .waveform_chain import _number
    snr = _number(settings, "snr_db", 35., -40., 200.)
    power = np.mean(abs(x)**2)*10**(-snr/10)
    rng = np.random.default_rng(seed)
    return x + np.sqrt(power/2)*(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))


def process_dual_stage(stage, inputs):
    from .waveform_chain import Wave, _number
    from .burst_link.channel.dual_polarization import DualPolarizationChannel
    from .burst_link.receive_filtering import RRCMatchedFilter
    from .burst_link.fastica_separator import FastICAPolarizationSeparator
    a, b = inputs
    if not np.isclose(a.fs, b.fs) or len(a.x) != len(b.x):
        raise ValueError("双路输入必须等长、采样率相同并共享时间轴")
    if not stage.enabled:
        return [Wave(w.x.copy(), w.fs, w.carrier_hz, w.start, w.stop, ("双路模块已旁路",), context=w.context)
                for w in inputs]
    p = stage.settings
    if stage.kind == "xpic":
        channel = DualPolarizationChannel(sample_rate_hz=a.fs, filter_taps=[1.],
            xpd_db=(_number(p, "xpd_a", 10., 0., 100.), _number(p, "xpd_b", 10., 0., 100.)),
            cross_phase_rad=(.3, -.4),
            source_delay_samples=(_number(p, "delay_a", 0., 0., 4096.), _number(p, "delay_b", 0., 0., 4096.)))
        x, y = channel(tensor(a.x), tensor(b.x))
        return [Wave(add_noise(z.numpy(), p, 3030+i), w.fs, w.carrier_hz, context=w.context,
                     notes=("双极化信道：两发射源经 2×2 交叉耦合进入共同接收窗口",))
                for i, (z, w) in enumerate(zip((x, y), inputs))]
    if any(w.context is None for w in inputs):
        raise ValueError("FastICA 需要两路带有已知同步头的帧信号")
    starts = []
    for w in inputs:
        matched = RRCMatchedFilter(w.context.shaper)(tensor(w.x)).numpy()
        start, _ = synchronize(Wave(matched, w.fs, context=w.context), .25)
        starts.append(start-w.context.shaper.delay_samples)
    separator = FastICAPolarizationSeparator(
        window_offset_samples=_number(p, "window_offset", 4500, 0, 200000, True),
        window_length_samples=_number(p, "window_length", 3000, 128, 200000, True))
    result = separator(tensor(a.x), tensor(b.x), start_a=starts[0], start_b=starts[1])
    note = (f"复数 FastICA 联合分离；迭代 {result.fastica.iterations} 次；"
            f"训练窗口 [{result.fit_start}, {result.fit_end})；路序区分度 {result.coefficient_margin:.4f}")
    return [Wave(z.numpy(), w.fs, w.carrier_hz, context=w.context, notes=(note,))
            for z, w in zip((result.waveform_a, result.waveform_b), inputs)]
