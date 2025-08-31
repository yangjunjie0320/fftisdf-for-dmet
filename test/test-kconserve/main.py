import os, sys, numpy, time
import pyscf, libdmet, fft, utils
from pyscf import lib

def get_kconserv_fast(cell, kpts, tol=1e-8, scale=10**8):
    nk = kpts.shape[0]
    a = cell.lattice_vectors() / (2*numpy.pi)     # (3,3)
    f = kpts @ a.T                              # 分数坐标 (nk,3)
    r = f - numpy.floor(f)                         # 规约到 [0,1)

    # 量化为 int 键，构造一个可二分查找的“字典表”
    q = int(scale)
    keys = numpy.rint(r * q).astype(numpy.int64)      # (nk,3)，范围 [0, q]
    # 极少数点可能在四舍五入后等于 q，把它绕回 0
    numpy.mod(keys, q, out=keys)

    dt = numpy.dtype([('x','<i8'), ('y','<i8'), ('z','<i8')])
    key_struct = keys.view(dt).ravel()          # (nk,)
    order = numpy.argsort(key_struct, kind='mergesort')
    key_sorted = key_struct[order]              # 排序后的键

    kconserv = numpy.empty((nk, nk, nk), dtype=numpy.int32)

    # 预先广播好的 l、m 分量，避免每次重建
    r_l = r[None, :, :]     # (1, nk, 3)
    r_m = r[:, None, :]     # (nk, 1, 3)

    for k in range(nk):
        # r_klm = (r[k] - r[l] + r[m]) mod 1
        tmp = r_m - r_l                       # (nk, nk, 3)
        tmp += r[k]                           # 逐元素 + r[k]
        tmp -= numpy.floor(tmp)                  # mod 1 到 [0,1)

        tgt_keys = numpy.rint(tmp * q).astype(numpy.int64).reshape(-1, 3)
        numpy.mod(tgt_keys, q, out=tgt_keys)

        tgt_struct = tgt_keys.view(dt).ravel()        # (nk*nk,)

        # 向量化二分查找 + 等值校验
        pos = numpy.searchsorted(key_sorted, tgt_struct)
        # 边界修正
        numpy.clip(pos, 0, nk-1, out=pos)

        hit = (key_sorted[pos] == tgt_struct)
        out = numpy.full(tgt_struct.shape, -1, dtype=numpy.int32)
        out[hit] = order[pos[hit]]                    # 排序位置 -> 原索引

        # 若存在极少因数值抖动造成 miss 的点，可尝试 pos-1 再次匹配
        if not numpy.all(hit):
            pos2 = numpy.clip(pos - 1, 0, nk-1)
            hit2 = ~hit & (key_sorted[pos2] == tgt_struct)
            out[hit2] = order[pos2[hit2]]
            hit |= hit2

        # 如果仍然有 miss，可以根据需要断言或放宽 scale
        # assert hit.all(), "Some (k,l,m) had no matching n; try larger scale."

        kconserv[k, :, :] = out.reshape(nk, nk)

    return kconserv

def main():
    from utils import build
    config = build()
    
    cell = config["cell"]
    # kmesh  = ['1-1-1', '1-1-2', '1-2-2', '2-2-2']
    kmesh  = ['2-2-3', '2-3-3', '3-3-3']
    kmesh += ['3-3-4', '3-4-4', '4-4-4']
    kmesh += ['4-4-6', '4-6-6', '6-6-6']

    for km in kmesh:
        m = [int(i) for i in km.split("-")]
        kpts = cell.make_kpts(m, wrap_around=True)
        nk = len(kpts)
        
        t0 = time.time()
        from fft.isdf import get_kconserv
        kconserv_ref = get_kconserv(cell, kpts)
        t1 = time.time()
        dt = (t1 - t0) / 3600
        print("nk = %4d, time_get_kconserv_ref = % 8.2e h" % (nk, dt), flush=True)

        t0 = time.time()
        kconserv_sol = get_kconserv_fast(cell, kpts)
        t1 = time.time()
        dt = (t1 - t0) / 3600
        print("nk = %4d, time_get_kconserv_fast = % 8.2e h" % (nk, dt), flush=True)

        assert numpy.allclose(kconserv_ref, kconserv_sol)

if __name__ == "__main__":
    main()
