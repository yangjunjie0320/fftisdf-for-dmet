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

    c = numpy.asarray(cc_obj.mo_coeff, dtype=numpy.complex128)
    eris = cc_obj.ao2mo(c)

    import os
    from pyscf.lib import chkfile
    if os.path.exists("t1_and_t2.chk"):
        print(f"load t1 and t2 from t1_and_t2.chk")
        t1 = chkfile.load("t1_and_t2.chk", "t1")
        t2 = chkfile.load("t1_and_t2.chk", "t2")
    else:
        print(f"get t1 and t2 from cc_obj")
        t1, t2 = cc_obj.get_init_guess(eris)

    t1 = numpy.asarray(t1, dtype=numpy.complex128)
    t2 = numpy.asarray(t2, dtype=numpy.complex128)
    cc_obj.kernel(t1=t1, t2=t2, eris=eris)

    t1 = cc_obj.t1
    t2 = cc_obj.t2
    print(f"t1.shape = {t1.shape}, t1.dtype = {t1.dtype}")
    print(f"t2.shape = {t2.shape}, t2.dtype = {t2.dtype}")
    print(f"save t1 and t2 to t1_and_t2.chk")
    chkfile.save("t1_and_t2.chk", "t1", t1)
    chkfile.save("t1_and_t2.chk", "t2", t2)

    ene_kccsd = cc_obj.e_tot
    ene_corr_kccsd = cc_obj.e_corr
    log.write("ene_kccsd = % 12.8f\n" % ene_kccsd)
    log.write("ene_corr_kccsd = % 12.8f\n" % ene_corr_kccsd)

    log.flush()

    # ene_corr_kccsd_t = cc_obj.ccsd_t(eris=eris)
    from pyscf.pbc.cc import kccsd_t_rhf, kccsd_t_rhf_slow
    ene_corr_kccsd_t = kccsd_t_rhf_slow.kernel(cc_obj, eris=eris, t1=t1, t2=t2)
    log.write("ene_corr_kccsd_t = % 12.8f\n" % ene_corr_kccsd_t)
    log.flush()

    # from pyscf.pbc.tools import k2gamma
    # mfg_obj = k2gamma.k2gamma(scf_obj)
    # mfg_obj.with_df = scf_obj.with_df

    # cc_obj = mfg_obj.CCSD()
    # cc_obj.verbose = 5
    # eris = cc_obj.ao2mo()
    # cc_obj.kernel(eris=eris)
    # log.write("ene_ccsd = % 12.8f\n" % cc_obj.e_tot)
    # log.write("ene_corr_ccsd = % 12.8f\n" % cc_obj.e_corr)
    # log.flush()

    # ene_corr_ccsd_t = cc_obj.ccsd_t(eris=eris)
    # log.write("ene_corr_ccsd_t = % 12.8f\n" % ene_corr_ccsd_t)
    # log.flush()

if __name__ == "__main__":
    main()
