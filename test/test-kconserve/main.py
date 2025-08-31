import os, sys, numpy, time
import pyscf, libdmet, fft, utils
from pyscf import lib

from line_profiler import profile
# from fft.isdf import get_kconserv
# get_kconserv = profile(get_kconserv)

@profile
def get_kconserv(cell, kpts):
    r'''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satisfies momentum conservation,

        (k(k) - k(l) + k(m) - k(n)) \dot a = 2n\pi

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2 * numpy.pi)

    k_dot_aT = lib.dot(kpts, a.T)
    kconserv = numpy.zeros((nkpts, nkpts, nkpts), dtype=int)
    kvKLM = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    kvKLM = kvKLM.reshape(-1, 3)
    kvKLM_dot_aT = lib.dot(kvKLM, a.T)

    kvN_dot_aT = lib.dot(kpts, a.T)

    for N in range(nkpts):
        kvKLMN = kvKLM_dot_aT - kvN_dot_aT[N]

        dkLMN = numpy.abs(kvKLMN - numpy.rint(kvKLMN))
        skLMN = numpy.sum(dkLMN, axis=-1)

        mkLMN = skLMN < 1e-9
        kconserv[mkLMN] = N

    return kconserv

def main():
    from utils import build
    config = build()
    
    cell = config["cell"]
    kmesh  = ['1-1-1', '1-1-2', '1-2-2', '2-2-2']
    kmesh += ['2-2-3', '2-3-3', '3-3-3']
    kmesh += ['3-3-4', '3-4-4', '4-4-4']
    kmesh += ['4-4-6', '4-6-6', '6-6-6']

    for km in kmesh:
        m = [int(i) for i in km.split("-")]
        kpts = cell.make_kpts(m, wrap_around=True)
        nk = len(kpts)
        
        t0 = time.time()
        kconserv = get_kconserv(cell, kpts)
        t1 = time.time()
        dt = (t1 - t0) / 3600
        print("nk = %4d, time_get_kconserv = % 8.2e h" % (nk, dt), flush=True)

if __name__ == "__main__":
    main()
