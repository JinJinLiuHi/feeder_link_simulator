"""Per-business-frame modems/codecs, using the BER module's exact definitions."""

from dataclasses import dataclass
import json

import torch

from link_budget.algorithm.ber_core.simulation import make_codec, make_modem
from .frame_config import code_dimensions, modulation_bits
from .burst_link.phase_tracking import PilotPhaseTracker


@dataclass
class BusinessFrame:
    mapper: object
    demapper: object
    encoder: object
    decoder: object
    k: int
    n: int
    m: int
    span: tuple


class MixedBurstFramer:
    def __init__(self, sync_symbols, header_symbols, plan, payload_symbols=None):
        self.sync_symbols = sync_symbols
        self.header_symbols = header_symbols
        self.sync_symbol_count = len(sync_symbols)
        self.header_symbol_count = len(header_symbols)
        self.payload_symbols = payload_symbols
        self.header_spans, self.codeword_spans, self.payload_spans, self.frames = [], [], [], []
        cursor = len(sync_symbols)
        codecs, modems = {}, {}
        for group in plan:
            self.header_spans.append((cursor, cursor+len(header_symbols)))
            cursor += len(header_symbols)
            payload_start = cursor
            spans = []
            for item in group:
                code, mod = item["code"], item["mod"]
                key = json.dumps(code, sort_keys=True)
                if key not in codecs:
                    codecs[key] = make_codec(code)
                encoder, decoder, k, n, _ = codecs[key]
                if (k, n) != code_dimensions(code):
                    raise ValueError(f"{code['name']} 的实际 K/N 与配置不一致")
                key = json.dumps(mod, sort_keys=True)
                if key not in modems:
                    if "legacy_name" in mod:
                        from .burst_link.modulation import create
                        from .burst_link.modulation.soft import soft_demodulate
                        legacy = create(mod["legacy_name"])
                        modems[key] = (legacy.modulate, lambda x, v, modem=legacy: soft_demodulate(x, modem, float(v)))
                    else:
                        mapper, demapper, _ = make_modem(mod)
                        modems[key] = mapper, demapper
                mapper, demapper = modems[key]
                m = modulation_bits(mod)
                length = (n+m-1)//m
                span = cursor, cursor+length
                spans.append(span)
                self.frames.append(BusinessFrame(mapper, demapper, encoder, decoder, k, n, m, span))
                cursor += length
            self.codeword_spans.append(tuple(spans))
            if payload_symbols is not None:
                if cursor-payload_start > payload_symbols:
                    raise ValueError("业务帧超过设置的业务数据符号数")
                cursor = payload_start+payload_symbols
            self.payload_spans.append((payload_start, cursor))
        self.capacity_bits = sum(f.k for f in self.frames)
        self.total_symbols = cursor
        self.phase_tracker = PilotPhaseTracker(sync_symbols=sync_symbols)
        self.last_phase_result = None

    @torch.no_grad()
    def __call__(self, info_bits):
        if info_bits.ndim != 1 or len(info_bits) != self.capacity_bits:
            raise ValueError(f"业务输入应为 {self.capacity_bits} bit")
        result = torch.zeros(self.total_symbols, dtype=torch.complex64)
        result[:self.sync_symbol_count] = self.sync_symbols
        for start, stop in self.header_spans:
            result[start:stop] = self.header_symbols
        cursor = 0
        for f in self.frames:
            word = info_bits[cursor:cursor+f.k].to(torch.float32).reshape(1, -1)
            coded = word if f.encoder is None else f.encoder(word)
            if coded.shape != (1, f.n):
                raise ValueError("编码器输出长度与业务帧配置不一致")
            padding = (-f.n) % f.m
            if padding:
                coded = torch.nn.functional.pad(coded, (0, padding))
            result[slice(*f.span)] = f.mapper(coded).reshape(-1)
            cursor += f.k
        return result

    @torch.no_grad()
    def decode(self, symbols):
        if symbols.ndim != 1 or len(symbols) != self.total_symbols:
            raise ValueError(f"接收突发应包含 {self.total_symbols} 个符号")
        tracked = self.phase_tracker(symbols, header_spans=self.header_spans, header_symbols=self.header_symbols)
        residual = torch.cat([tracked.symbols[a:b]-self.header_symbols for a, b in self.header_spans])
        variance = max(float(residual.abs().square().mean()), 1e-8)
        outputs = []
        for f in self.frames:
            data = tracked.symbols[slice(*f.span)].to(torch.complex64).reshape(1, -1)
            llr = f.demapper(data, torch.tensor(variance)).reshape(1, -1)[:, :f.n]
            bits = llr > 0 if f.decoder is None else f.decoder(llr)
            if bits.shape != (1, f.k):
                raise ValueError("译码器输出长度与业务帧配置不一致")
            outputs.append(bits.reshape(-1).to(torch.uint8))
        self.last_phase_result = tracked
        return torch.cat(outputs)
