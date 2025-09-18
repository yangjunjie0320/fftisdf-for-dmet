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
    scf_obj = config["mf"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krhf = scf_obj.kernel(dm0)
    scf_obj.analyze()
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    # kmesh = config["kmesh"]
    # print("kmesh = ", kmesh)
    # if nkpt <= 8:
    #     from pyscf.pbc.tools.k2gamma import k2gamma
    #     mfg = k2gamma(scf_obj, kmesh)
    #     print("mfg = ", mfg, type(mfg))

    #     t0 = time.time()
    #     import pyscf.pbc.mp.mp2
    #     mp_obj = mfg.MP2()        
    #     print("mp_obj = ", mp_obj, type(mp_obj))

    #     mp_obj.verbose = 10
    #     mp_obj.kernel(with_t2=False)
    #     log.write("time_kmp2 = % 6.2f\n" % (time.time() - t0))
    #     log.write("ene_kmp2 = % 12.8f\n" % mp_obj.e_tot)
    #     log.write("ene_corr_kmp2 = % 12.8f\n" % mp_obj.e_corr)
    #     log.write("ene_corr_os = % 12.8f\n" % mp_obj.e_corr_os)
    #     log.flush()

    #     import pyscf.pbc.cc.ccsd
    #     cc_obj = mfg.CCSD()
    #     print("cc_obj = ", cc_obj, type(cc_obj))
    #     cc_obj.verbose = 10
    #     cc_obj.kernel()
    #     log.write("time_kccsd = % 6.2f\n" % (time.time() - t0))
    #     log.write("ene_kccsd = % 12.8f\n" % cc_obj.e_tot)
    #     log.write("ene_corr_kccsd = % 12.8f\n" % cc_obj.e_corr)
    #     log.flush()

    # # from pyscf.pbc.cc import KCCSD
    # # cc_obj = KCCSD(scf_obj)
    # # cc_obj.verbose = 10
    # # eris = cc_obj.ao2mo()
    # # cc_obj.kernel(eris=eris)
    # # ene_kccsd = cc_obj.e_tot
    # # ene_corr_kccsd = cc_obj.e_corr
    # # log.write("ene_kccsd = % 12.8f\n" % ene_kccsd)
    # # log.write("ene_corr_kccsd = % 12.8f\n" % ene_corr_kccsd)
    # # log.flush()

if __name__ == "__main__":
    main()
