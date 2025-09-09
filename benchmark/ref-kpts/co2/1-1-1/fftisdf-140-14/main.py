import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main():
    from utils import build
    config = build()

    cell = config["cell"]
    df_obj = config["df"]
    kpts = config["kpts"]
    nkpt = nimg = len(kpts)

    dm0 = config["dm0"]
    nao = cell.nao_nr()
    natm = cell.natm

    log = open("out.log", "w")
    log.write("method = %s\n" % config["density_fitting_method"])
    log.write("basis = %s\n" % config["basis"])
    log.write("natm = %d\n" % natm)
    log.write("nkpt = %d\n" % nkpt)
    log.write("nao = %d\n" % nao)
    log.flush()

    t0 = time.time()
    df_obj.build()
    log.write("time_build_df = % 6.2f\n" % (time.time() - t0))

    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()
    if naux is not None:
        log.write("naux = %d\n" % naux)

    t0 = time.time()
    scf_obj = pyscf.pbc.dft.KRKS(cell, kpts)
    scf_obj.xc = "pbe"
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krks = scf_obj.kernel(dm0)
    log.write("time_krks = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krks = % 12.8f\n" % ene_krks)
    log.flush()

    t0 = time.time()
    scf_obj = config["mf"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krhf = scf_obj.kernel(dm0)
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    t0 = time.time()
    from pyscf.pbc.mp import KMP2
    mp_obj = KMP2(scf_obj)
    mp_obj.verbose = 10
    mp_obj.kernel()
    log.write("time_kmp2 = % 6.2f\n" % (time.time() - t0))
    log.write("ene_kmp2 = % 12.8f\n" % mp_obj.e_tot)
    log.write("ene_corr_kmp2 = % 12.8f\n" % mp_obj.e_corr)
    log.write("ene_corr_os = % 12.8f\n" % mp_obj.e_corr_os)
    log.flush()

    from pyscf.pbc.cc import KCCSD
    cc_obj = KCCSD(scf_obj)
    cc_obj.verbose = 10
    eris = cc_obj.ao2mo()
    t1, t2 = cc_obj.get_init_guess(eris)
    t1 = numpy.asarray(t1, dtype=numpy.complex128)
    t2 = numpy.asarray(t2, dtype=numpy.complex128)
    cc_obj.kernel(t1=t1, t2=t2, eris=eris)
    ene_kccsd = cc_obj.e_tot
    ene_corr_kccsd = cc_obj.e_corr
    log.write("ene_kccsd = % 12.8f\n" % ene_kccsd)
    log.write("ene_corr_kccsd = % 12.8f\n" % ene_corr_kccsd)
    log.flush()

if __name__ == "__main__":
    main()
