from __future__ import annotations

from numbers import Integral

import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder

from ._validation import resolve_device, validate_information_words


class NRLDPCEncoder(LDPC5GEncoder):
    """NR LDPC encoder backed by Sionna, including its rate-matching operation.

    ``k`` is the input length and ``n`` is the final transmitted length.  This
    class does not insert a transport CRC, split a burst, or perform modulation.
    """

    def __init__(self, k: int, n: int, device: str | torch.device = "cpu"):
        if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, Integral) or not isinstance(n, Integral):
            raise TypeError("NR LDPC 的 k 和 n 必须是整数")
        resolved = resolve_device(device)
        super().__init__(int(k), int(n), device=str(resolved), precision="single")
        self.information_bits_per_codeword = int(self.k)
        self.encoded_bits_per_codeword = int(self.n)

    def encode(self, info_words: torch.Tensor) -> torch.Tensor:
        target = resolve_device(self.device)
        validate_information_words(info_words, self.information_bits_per_codeword, target)
        return self(info_words.to(dtype=torch.float32)).to(dtype=torch.uint8)
