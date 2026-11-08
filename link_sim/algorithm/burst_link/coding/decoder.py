"""Create a matching Sionna decoder for a bundled LDPC encoder."""
from __future__ import annotations
import operator
import torch
from sionna.phy.fec.ldpc import LDPC5GDecoder
from .nr_ldpc import NRLDPCEncoder

class LDPCDecoder:
    """Adapter accepting log(P(1)/P(0)) and outputting K information bits."""
    def __init__(self, encoder: object, *, num_iter: int = 20):
        if isinstance(num_iter, bool):
            raise TypeError("num_iter 必须是正整数")
        try:
            num_iter = operator.index(num_iter)
        except TypeError as exc:
            raise TypeError("num_iter 必须是正整数") from exc
        if num_iter <= 0:
            raise ValueError("num_iter 必须是正整数")
        self.information_bits_per_codeword = encoder.information_bits_per_codeword
        self.encoded_bits_per_codeword = encoder.encoded_bits_per_codeword
        self.device = torch.device(encoder.device)
        kwargs = dict(hard_out=True, num_iter=num_iter, cn_update="boxplus-phi",
                      device=str(self.device), precision="single")
        if isinstance(encoder, NRLDPCEncoder):
            self._decoder = LDPC5GDecoder(encoder, return_infobits=True, **kwargs)
        else:
            raise TypeError("当前组帧链路使用 NR LDPC 编码器")
        self.num_iter = num_iter

    @torch.no_grad()
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits [E] or [W,E] into uint8 bits [K] or [W,K]."""
        if not isinstance(logits, torch.Tensor):
            raise TypeError("译码输入必须是 torch.Tensor")
        if logits.ndim not in (1, 2) or logits.is_complex():
            raise ValueError("译码输入必须是一维或二维实数 logits")
        if logits.shape[-1] != self.encoded_bits_per_codeword:
            raise ValueError("译码输入长度必须与编码器输出长度一致")
        if logits.device != self.device:
            raise ValueError("译码输入与译码器必须在同一设备")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("译码输入不能包含非有限值")
        if logits.ndim == 2 and logits.shape[0] == 0:
            return torch.empty((0, self.information_bits_per_codeword),
                               dtype=torch.uint8, device=self.device)
        original_1d = logits.ndim == 1
        data = logits.unsqueeze(0) if original_1d else logits
        decoded = self._decoder(data.to(torch.float32))
        decoded = decoded[..., :self.information_bits_per_codeword].to(torch.uint8)
        return decoded.squeeze(0) if original_1d else decoded

def create_decoder(encoder: object, *, num_iter: int = 20) -> LDPCDecoder:
    """Create a decoder matching the supplied bundled encoder instance."""
    return LDPCDecoder(encoder, num_iter=num_iter)
