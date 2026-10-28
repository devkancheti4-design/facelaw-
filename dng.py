"""LAW 1 FROM A REAL PHONE FILE: a minimal DNG reader (pure Python + numpy, no RAW library).
Reads the raw CFA sub-image (uncompressed 8/16-bit, bit-packed 10/12/14-bit, or lossless JPEG = compression 7) and the
tags Law 1 needs, then returns a photon map:
    b = BlackLevel, W = WhiteLevel, CFAPattern, NoiseProfile (S, O) per colour plane  (variance = S x + O, x normalised)
    Law 1:  x = (DN - b) / (W - b),  Var(DN) = g (DN - b) + g^2 s_r^2   =>   g = S (W - b)  [DN per photon],
            s_r^2 = O / S^2  [e-^2]  -- the phone maker's own photon-transfer measurement, nothing tuned
    N^ = (DN - b) / g per CFA pixel;  2x2 Bayer blocks summed -> a half-resolution photon map with exact variance.
Also a writer for test files (uncompressed or lossless-JPEG), so the reader is proven round-trip exact."""
import struct
import numpy as np

TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1), 7: ("B", 1), 8: ("h", 2),
         9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8), 13: ("I", 4), 16: ("Q", 8)}


# ------------------------------------------------------------------------------------------------ TIFF / DNG ---
def _ifd(buf, off, e):
    n = struct.unpack(e + "H", buf[off:off + 2])[0]
    tags = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack(e + "HHI4s", buf[off + 2 + 12 * i: off + 14 + 12 * i])
        if typ not in TYPES:
            continue
        f, sz = TYPES[typ]
        size = sz * cnt
        data = val if size <= 4 else buf[struct.unpack(e + "I", val)[0]: struct.unpack(e + "I", val)[0] + size]
        if typ == 2:
            tags[tag] = data[:cnt].split(b"\0")[0].decode("latin-1")
        elif typ in (5, 10):
            v = struct.unpack(e + f[0] * (2 * cnt), data[:size])
            tags[tag] = [v[2 * k] / v[2 * k + 1] if v[2 * k + 1] else 0.0 for k in range(cnt)]
        else:
            tags[tag] = list(struct.unpack(e + f * cnt, data[:size]))
    nxt = struct.unpack(e + "I", buf[off + 2 + 12 * n: off + 6 + 12 * n])[0]
    return tags, nxt


def _all_ifds(buf, e, off, seen=None):
    seen = seen if seen is not None else set()
    out = []
    while off and off not in seen and off < len(buf):
        seen.add(off)
        tags, nxt = _ifd(buf, off, e)
        out.append(tags)
        for sub in tags.get(330, []):                         # SubIFDs
            out += _all_ifds(buf, e, sub, seen)
        off = nxt
    return out


def _unpack_bits(raw, bits, n):
    """big-endian bit packing (DNG rule for BitsPerSample not 8/16/32)"""
    a = np.unpackbits(np.frombuffer(raw, np.uint8))[: n * bits].reshape(n, bits)
    return (a * (1 << np.arange(bits - 1, -1, -1))).sum(1)


# ------------------------------------------------------------------------------------------ lossless JPEG ---
def ljpeg_decode(data):
    """ITU T.81 process 14 (SOF3).  -> (rows, cols * components) array, components interleaved along each row"""
    pos, huff, restart = 2, {}, 0
    while True:
        assert data[pos] == 0xFF, "bad marker"
        m = data[pos + 1]; L = struct.unpack(">H", data[pos + 2:pos + 4])[0]; seg = data[pos + 4:pos + 2 + L]
        if m == 0xC3:
            P, Y, X, Nf = seg[0], struct.unpack(">H", seg[1:3])[0], struct.unpack(">H", seg[3:5])[0], seg[5]
        elif m == 0xC4:
            k = 0
            while k < len(seg):
                tc_th = seg[k]; counts = seg[k + 1:k + 17]; vals = seg[k + 17:k + 17 + sum(counts)]
                code, codes, vi = 0, {}, 0
                for length in range(1, 17):
                    for _ in range(counts[length - 1]):
                        codes[(length, code)] = vals[vi]; vi += 1; code += 1
                    code <<= 1
                huff[tc_th & 15] = codes
                k += 17 + sum(counts)
        elif m == 0xDD:
            restart = struct.unpack(">H", seg[:2])[0]
        elif m == 0xDA:
            Ns = seg[0]; sel = [seg[2 + 2 * i] >> 4 for i in range(Ns)]; Ss, Pt = seg[1 + 2 * Ns], seg[3 + 2 * Ns] & 15
            pos += 2 + L
            break
        pos += 2 + L
    if restart:
        raise NotImplementedError("lossless JPEG with restart intervals is not supported yet")
    # entropy-coded data: unstuff FF00, stop at the next marker
    ent = bytearray(); i = pos
    while i < len(data):
        b = data[i]
        if b == 0xFF:
            if data[i + 1] == 0x00:
                ent.append(0xFF); i += 2; continue
            break
        ent.append(b); i += 1
    ent += b"\0" * 8
    lut = {}
    for t, tab in huff.items():                                    # 16-bit peek -> (length, symbol)
        L16 = np.zeros(65536, np.int32)
        for (l, code), v in tab.items():
            L16[code << (16 - l):(code + 1) << (16 - l)] = (l << 8) | v
        lut[t] = L16.tolist()
    state = [0, 0, 0]                                              # acc, bits in acc, next byte

    def need(n):
        acc, nb, k = state
        while nb < n:
            acc = ((acc << 8) | ent[k]) & 0xFFFFFFFFFFFFFFFF; nb += 8; k += 1
        state[0], state[1], state[2] = acc, nb, k

    def getbits(n):
        if n == 0:
            return 0
        need(n)
        acc, nb, k = state
        v = (acc >> (nb - n)) & ((1 << n) - 1)
        state[1] = nb - n
        return v

    def diff(t):
        need(16)
        acc, nb, k = state
        e = lut[t][(acc >> (nb - 16)) & 0xFFFF]
        l, s = e >> 8, e & 0xFF
        if l == 0:
            raise ValueError("bad Huffman code")
        state[1] = nb - l
        if s == 0:
            return 0
        if s == 16:
            return 32768
        v = getbits(s)
        return v - ((1 << s) - 1) if v < (1 << (s - 1)) else v

    out = np.zeros((Y, X * Nf), np.int64)
    init = 1 << (P - Pt - 1)
    for r in range(Y):
        for c in range(X):
            for k in range(Nf):
                col = c * Nf + k
                d = diff(sel[k])
                if r == 0 and c == 0:
                    pred = init
                elif r == 0:
                    pred = out[r, col - Nf]
                elif c == 0:
                    pred = out[r - 1, col]
                else:
                    a, bb, cc = out[r, col - Nf], out[r - 1, col], out[r - 1, col - Nf]
                    pred = {1: a, 2: bb, 3: cc, 4: a + bb - cc, 5: a + ((bb - cc) >> 1), 6: bb + ((a - cc) >> 1),
                            7: (a + bb) >> 1}[Ss]
                out[r, col] = (pred + d) & 0xFFFF
    return out


def ljpeg_encode(img, P=16, Nf=2):
    """predictor 1 (left; first column from above), one canonical Huffman table built from the data"""
    Y, XN = img.shape; X = XN // Nf
    img = img.astype(np.int64)
    pred = np.empty_like(img)
    pred[0, :Nf] = 1 << (P - 1); pred[0, Nf:] = img[0, :-Nf]; pred[1:, :Nf] = img[:-1, :Nf]; pred[1:, Nf:] = img[1:, :-Nf]
    d = (img - pred)
    d = np.where(d > 32768, d - 65536, np.where(d < -32767, d + 65536, d))
    ssss = np.where(d == 0, 0, np.floor(np.log2(np.abs(d) + 0.5)).astype(int) + 1)
    ssss = np.where(d == 32768, 16, ssss)
    freq = np.bincount(ssss.ravel(), minlength=17)
    syms = [s for s in np.argsort(-freq) if freq[s] > 0]
    lengths = {}
    import heapq
    heap = [(int(freq[s]), i, [s]) for i, s in enumerate(syms)]
    heapq.heapify(heap)
    if len(heap) == 1:
        lengths[syms[0]] = 1
    while len(heap) > 1:
        f1, i1, s1 = heapq.heappop(heap); f2, i2, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] = lengths.get(s, 0) + 1
        heapq.heappush(heap, (f1 + f2, min(i1, i2), s1 + s2))
    assert max(lengths.values()) <= 16
    order = sorted(syms, key=lambda s: (lengths[s], s))
    codes, code, prev = {}, 0, lengths[order[0]]
    for s in order:
        code <<= lengths[s] - prev; prev = lengths[s]
        codes[s] = (lengths[s], code); code += 1
    counts = [sum(1 for s in order if lengths[s] == l) for l in range(1, 17)]
    bits = []
    for v, s in zip(d.ravel(), ssss.ravel()):
        l, c = codes[int(s)]
        bits.append((c, l))
        if 0 < s < 16:
            vv = int(v) if v > 0 else int(v) + (1 << int(s)) - 1
            bits.append((vv, int(s)))
    acc = 0; nb = 0
    for c, l in bits:
        acc = (acc << l) | c; nb += l
    pad = (-nb) % 8
    acc = (acc << pad) | ((1 << pad) - 1); nb += pad
    raw = acc.to_bytes(nb // 8, "big").replace(b"\xFF", b"\xFF\x00")
    sof = struct.pack(">BHHB", P, Y, X, Nf) + b"".join(struct.pack(">BBB", k + 1, 0x11, 0) for k in range(Nf))
    dht = bytes([0x00]) + bytes(counts) + bytes(order)
    sos = bytes([Nf]) + b"".join(bytes([k + 1, 0x00]) for k in range(Nf)) + bytes([1, 0, 0])
    seg = lambda m, b: b"\xFF" + bytes([m]) + struct.pack(">H", len(b) + 2) + b
    return b"\xFF\xD8" + seg(0xC3, sof) + seg(0xC4, dht) + seg(0xDA, sos) + raw + b"\xFF\xD9"


# ---------------------------------------------------------------------------------------------- the reader ---
def read(path):
    buf = open(path, "rb").read()
    e = "<" if buf[:2] == b"II" else ">"
    assert struct.unpack(e + "H", buf[2:4])[0] == 42, "not a TIFF/DNG file"
    ifds = _all_ifds(buf, e, struct.unpack(e + "I", buf[4:8])[0])
    raw = next(t for t in ifds if t.get(262, [0])[0] in (32803, 34892) and t.get(254, [0])[0] == 0)
    W_, H_ = raw[256][0], raw[257][0]; bits = raw[258][0]; comp = raw.get(259, [1])[0]
    img = np.zeros((H_, W_), np.int64)
    if 324 in raw:                                                  # tiles
        tw, tl = raw[322][0], raw[323][0]
        across = (W_ + tw - 1) // tw
        for k, (o, n) in enumerate(zip(raw[324], raw[325])):
            r0, c0 = (k // across) * tl, (k % across) * tw
            tile = _decode_block(buf[o:o + n], comp, bits, tl, tw, e)
            h, w = min(tl, H_ - r0), min(tw, W_ - c0)
            img[r0:r0 + h, c0:c0 + w] = tile[:h, :w]
    else:                                                           # strips
        rps = raw.get(278, [H_])[0]
        for k, (o, n) in enumerate(zip(raw[273], raw[279])):
            r0 = k * rps; h = min(rps, H_ - r0)
            img[r0:r0 + h] = _decode_block(buf[o:o + n], comp, bits, h, W_, e)[:h]
    lin = raw.get(50712) or next((t.get(50712) for t in ifds if 50712 in t), None)
    if lin:
        img = np.array(lin)[np.clip(img, 0, len(lin) - 1)]
    meta = {}
    for t in ifds:                                                  # tags may sit in IFD0 or in the raw IFD
        meta.update({k: v for k, v in t.items() if k in (50714, 50713, 50717, 51041, 33421, 33422, 50728, 50730, 34855, 33434, 271, 272)})
    meta.update({k: v for k, v in raw.items() if k in (50714, 50713, 50717, 51041, 33421, 33422)})
    return img, meta


def _decode_block(data, comp, bits, h, w, e):
    if comp == 1:
        if bits in (8, 16):
            a = np.frombuffer(data[:h * w * bits // 8], (e.replace("<", "<") + ("u1" if bits == 8 else "u2")))
            return a.reshape(h, w).astype(np.int64)
        return _unpack_bits(data, bits, h * w).reshape(h, w)
    if comp == 7:
        d = ljpeg_decode(data)
        return d.reshape(h, w) if d.size == h * w else d[:h, :w]
    raise NotImplementedError("compression %d" % comp)


def law1(img, meta):
    """Law 1 parameters straight from the file -> photon estimate per CFA pixel and the 2x2 photon map"""
    pat_dim = meta.get(33421, [2, 2]); pat = meta.get(33422, [0, 1, 1, 2])
    bl = meta.get(50714, [0]); bdim = meta.get(50713, [1, 1])
    W = meta.get(50717, [int(img.max())])[0]
    plane = np.array(pat).reshape(pat_dim[0], pat_dim[1])
    rr, cc = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    cfa = plane[rr % pat_dim[0], cc % pat_dim[1]]
    b = np.array(bl).reshape(bdim[0], bdim[1])[rr % bdim[0], cc % bdim[1]] if len(bl) == bdim[0] * bdim[1] else np.full(img.shape, bl[0])
    npf = meta.get(51041)
    if not npf:
        raise ValueError("no NoiseProfile tag: Law 1's gain must then be measured (photon-transfer on flat patches)")
    S = np.array(npf[0::2]); O = np.array(npf[1::2])
    Sp, Op = (S[cfa], O[cfa]) if len(S) > 1 else (np.full(img.shape, S[0]), np.full(img.shape, O[0]))
    g = Sp * (W - b)                                        # DN per photon
    sr2 = Op / Sp ** 2                                      # read noise, e-^2
    Nh = (img - b) / g
    h2, w2 = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    photon = Nh[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
    var = (np.maximum(Nh, 0) + sr2)[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
    return dict(Nh=Nh, photon=photon, var=var, g=float(np.median(g)), sr=float(np.sqrt(np.median(sr2))),
                black=float(np.median(b)), white=W, saturated=img >= W)


# ---------------------------------------------------------------------------------------------- the writer ---
def write(path, img, black, white, S, O, pattern=(0, 1, 1, 2), compression=1, tile=None):
    """a minimal valid DNG: IFD0 is the raw CFA image (16-bit), tags for Law 1; compression 1 or 7 (lossless JPEG)"""
    H_, W_ = img.shape
    if compression == 7:
        tw, tl = tile or (W_, H_)
        blocks, offs = [], []
        for r0 in range(0, H_, tl):
            for c0 in range(0, W_, tw):
                t = np.zeros((tl, tw), np.int64); part = img[r0:r0 + tl, c0:c0 + tw]; t[:part.shape[0], :part.shape[1]] = part
                blocks.append(ljpeg_encode(t.reshape(tl, tw), P=16, Nf=2))
    else:
        blocks = [img.astype("<u2").tobytes()]
    entries = []
    def ent(tag, typ, vals):
        entries.append((tag, typ, vals))
    ent(254, 4, [0]); ent(256, 4, [W_]); ent(257, 4, [H_]); ent(258, 3, [16]); ent(259, 3, [compression])
    ent(262, 3, [32803]); ent(277, 3, [1])
    if compression == 7:
        ent(322, 4, [tw]); ent(323, 4, [tl]); ent(324, 4, [0] * len(blocks)); ent(325, 4, [len(b) for b in blocks])
    else:
        ent(273, 4, [0]); ent(278, 4, [H_]); ent(279, 4, [len(blocks[0])])
    ent(33421, 3, [2, 2]); ent(33422, 1, list(pattern)); ent(50706, 1, [1, 4, 0, 0])
    ent(50714, 4, [int(black)]); ent(50717, 4, [int(white)])
    ent(51041, 12, [v for s, o in zip(S, O) for v in (s, o)])
    entries.sort()
    fmt = {1: "B", 3: "H", 4: "I", 12: "d"}; size = {1: 1, 3: 2, 4: 4, 12: 8}
    ifd_off = 8; n = len(entries)
    data_off = ifd_off + 2 + 12 * n + 4
    extra = b""; recs = []
    blob_off = None
    for tag, typ, vals in entries:
        raw = struct.pack("<" + fmt[typ] * len(vals), *vals)
        if len(raw) <= 4:
            recs.append(struct.pack("<HHI", tag, typ, len(vals)) + raw.ljust(4, b"\0"))
        else:
            recs.append((tag, typ, len(vals), len(extra)))
            extra += raw + (b"\0" if len(raw) % 2 else b"")
    pix_off = data_off + len(extra)
    offsets, o = [], pix_off
    for b in blocks:
        offsets.append(o); o += len(b)
    out = []
    for r, (tag, typ, vals) in zip(recs, entries):
        if isinstance(r, tuple):
            out.append(struct.pack("<HHII", r[0], r[1], r[2], data_off + r[3]))
        else:
            if tag in (273, 324) and len(vals) == 1:
                r = struct.pack("<HHI", tag, typ, 1) + struct.pack("<I", offsets[0])
            out.append(r)
    # tile offsets stored out of line: patch them
    extra = bytearray(extra)
    for (tag, typ, vals), r in zip(entries, recs):
        if tag == 324 and isinstance(r, tuple):
            extra[r[3]:r[3] + 4 * len(vals)] = struct.pack("<" + "I" * len(vals), *offsets)
    with open(path, "wb") as f:
        f.write(b"II" + struct.pack("<HI", 42, ifd_off) + struct.pack("<H", n) + b"".join(out) + struct.pack("<I", 0)
                + bytes(extra) + b"".join(blocks))
