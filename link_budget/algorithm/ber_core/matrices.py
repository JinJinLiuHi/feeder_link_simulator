"""Strict matrix readers and binary systematic encoding for local LDPC data."""

import re
import numpy as np
from scipy import sparse


def _numbers(line):
    raw = line.split("#", 1)[0].strip()
    if not raw:
        return []
    if "," in raw and any(not part.strip() for part in raw.split(",")):
        raise ValueError("矩阵中包含空的 CSV 字段")
    tokens = re.split(r"[\s,]+", raw)
    if any(not re.fullmatch(r"-?[0-9]+", token) for token in tokens):
        raise ValueError("矩阵只能包含整数")
    return list(map(int, tokens))


def read_matrix(text, kind, z=1, direction=1):
    if len(text.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("矩阵文本超过 16 MiB")
    if kind == "alist":
        lines = [
            line for line in text.lstrip("\ufeff").splitlines() if not line.lstrip().startswith("#")
        ]
        if len(lines) < 4:
            raise ValueError("alist 文件缺少头部")
        head = _numbers(lines[0])
        degrees = _numbers(lines[1])
        if len(head) != 2 or len(degrees) != 2:
            raise ValueError("alist 头部应为 N M 及最大列度/行度")
        n, m = head
        dc, dr = degrees
        while len(lines) > 4 + n + m and not lines[-1].strip():
            lines.pop()
        if not 0 < m < n <= 100000 or m > 100000 or dc <= 0 or dr <= 0:
            raise ValueError("alist 维度或最大度数无效")
        cols = _numbers(lines[2])
        rows = _numbers(lines[3])
        if len(cols) != n or len(rows) != m or len(lines) != 4 + n + m:
            raise ValueError("alist 维度、度数数量或邻接行数不匹配")
        if (
            min(cols + rows) < 0
            or max(cols) != dc
            or max(rows) != dr
            or sum(cols) != sum(rows)
            or sum(rows) > 1000000
        ):
            raise ValueError("alist 度数声明不一致或边数超限")
        col_edges = set()
        row_edges = set()
        for index, degree in enumerate(cols):
            values = _numbers(lines[4 + index])
            valid = [v for v in values if v]
            if (
                len(values) not in (degree, dc)
                or len(valid) != degree
                or len(set(valid)) != degree
                or any(v < 1 or v > m for v in valid)
            ):
                raise ValueError(f"alist 第 {index+1} 列邻接表无效")
            col_edges.update((v - 1, index) for v in valid)
        for index, degree in enumerate(rows):
            values = _numbers(lines[4 + n + index])
            valid = [v for v in values if v]
            if (
                len(values) not in (degree, dr)
                or len(valid) != degree
                or len(set(valid)) != degree
                or any(v < 1 or v > n for v in valid)
            ):
                raise ValueError(f"alist 第 {index+1} 行邻接表无效")
            row_edges.update((index, v - 1) for v in valid)
        if row_edges != col_edges:
            raise ValueError("alist 行邻接与列邻接描述不一致")
        rr, cc = zip(*sorted(row_edges)) if row_edges else ([], [])
        return sparse.csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(m, n))
    lines = [line for line in text.lstrip("\ufeff").splitlines() if line.split("#", 1)[0].strip()]
    if kind == "qc" and lines and lines[0].strip().upper().startswith("Z"):
        header = lines.pop(0).split()
        if len(header) != 2 or int(header[1]) != z:
            raise ValueError("文件中的 Z 声明与输入提升因子不一致")
    values = [_numbers(line) for line in lines]
    if not values or not values[0] or any(len(row) != len(values[0]) for row in values):
        raise ValueError("矩阵为空或每行列数不一致")
    if kind == "h":
        if (
            len(values) > 16384
            or len(values[0]) > 16384
            or any(v not in (0, 1) for row in values for v in row)
        ):
            raise ValueError("H 必须为二元矩阵，行列均不超过 16384")
        return sparse.csr_matrix(np.array(values, dtype=np.int8))
    if kind != "qc" or type(z) is not int or not 1 <= z <= 8192 or direction not in (-1, 1):
        raise ValueError("QC 提升因子或循环方向无效")
    if (
        len(values) * z > 16384
        or len(values[0]) * z > 16384
        or any(v < -1 or v >= z for row in values for v in row)
    ):
        raise ValueError("QC 维度超限或移位值超出 -1～Z−1")
    rr = []
    cc = []
    offset = np.arange(z)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v >= 0:
                rr.extend((i * z + offset).tolist())
                cc.extend((j * z + (offset + direction * v) % z).tolist())
    if len(rr) > 1000000:
        raise ValueError("QC 展开边数超过 100 万")
    return sparse.csr_matrix(
        (np.ones(len(rr), np.int8), (rr, cc)), shape=(len(values) * z, len(values[0]) * z)
    )


def systematic_form(h):
    """Eliminate over GF(2), preserving redundant original checks in the decoder."""
    m, n = h.shape
    if n > 16384 or m > 16384 or h.nnz > 1000000:
        raise ValueError("自定义矩阵规模超出限制")
    basis = {}
    for i in range(m):
        value = sum(1 << int(c) for c in h.indices[h.indptr[i] : h.indptr[i + 1]])
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    parity = sorted(basis)
    parity_set = set(parity)
    info = [c for c in range(n) if c not in parity_set]
    if not info or not parity or len(info) * len(parity) > 32000000:
        raise ValueError("矩阵没有有效信息位/校验约束，或编码关系超过内存限制")
    solved = {}
    relation = np.zeros((len(parity), len(info)), np.float32)
    for i, pivot in enumerate(parity):
        value = basis[pivot] ^ (1 << pivot)
        for prior in parity[:i]:
            if value & (1 << prior):
                value ^= (1 << prior) | solved[prior]
        solved[pivot] = value
        packed = np.unpackbits(
            np.frombuffer(value.to_bytes((n + 7) // 8, "little"), np.uint8), bitorder="little"
        )
        relation[i] = packed[info]
    permutation = np.array(info + parity)
    return h[:, permutation].tocsr(), relation, permutation


def ldpc_codec(h, device, iterations, staircase=False):
    import torch
    from sionna.phy.fec.ldpc import LDPCBPDecoder

    n = h.shape[1]
    if staircase:
        k = n - h.shape[0]
        if k <= 0:
            raise ValueError("DVB 矩阵维度无效")
        hp = h[:, k:].tocsr()
        expected = sparse.eye(h.shape[0], format="csr", dtype=np.int8) + sparse.diags(
            np.ones(h.shape[0] - 1, np.int8), -1, format="csr", dtype=np.int8
        )
        if (hp != expected).nnz:
            raise ValueError("DVB 矩阵不符合双对角阶梯校验结构")
        a = h[:, :k].tocoo()
        rows = torch.tensor(a.row.astype(np.int64), device=device)
        cols = torch.tensor(a.col.astype(np.int64), device=device)

        def encode(u):
            sums = torch.zeros((u.shape[0], h.shape[0]), dtype=torch.float32, device=device)
            sums.index_add_(1, rows, u[:, cols])
            parity = torch.cumsum(sums, dim=1).remainder(2)
            return torch.cat((u, parity), dim=1)

    else:
        h, relation, _ = systematic_form(h)
        k = relation.shape[1]
        relation = torch.tensor(relation.T, device=device)

        def encode(u):
            return torch.cat((u, torch.matmul(u, relation).remainder(2)), dim=1)

    padded = h.shape[0] * int(np.diff(h.indptr).max()) + h.shape[1] * int(
        np.asarray(h.sum(axis=0)).max()
    )
    if padded > 4000000:
        raise ValueError("译码节点规模超过内存限制")
    decoder = LDPCBPDecoder(
        h, hard_out=True, num_iter=iterations, precision="single", device=device
    )
    return encode, lambda llr: decoder(llr)[..., :k], k, n, h
