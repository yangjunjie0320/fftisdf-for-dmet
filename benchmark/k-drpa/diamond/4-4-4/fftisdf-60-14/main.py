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
    log.flush()

    t0 = time.time()
    scf_obj = config["mf"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krhf = scf_obj.kernel(dm0)
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    kconserv3 = df_obj.kconserv3
    print("kconserv3 = ", kconserv3.shape)

    def get_kconserv_new(*args, **kwargs):
        print("args = ", args)
        print("kwargs = ", kwargs)
        assert isinstance(df_obj, fft.ISDF)
        assert df_obj._kconserv3 is not None
        res = df_obj.kconserv3
        return res
    pyscf.pbc.lib.kpts_helper.get_kconserv = get_kconserv_new

    from pyscf.pbc.lib import kpts_helper
    kpts_helper_old = kpts_helper.KptsHelper
    def kpts_helper_new(*args, **kwargs):
        kwargs["init_symm_map"] = False
        res = kpts_helper_old(*args, **kwargs)
        return res
    kpts_helper.KptsHelper = kpts_helper_new

    mp = pyscf.pbc.mp.KMP2(scf_obj)
    mp.verbose = 10

    for nw in [20, 25, 30, 35, 40]:
        from krpa import krpa_pol_with_isdf, krpa_corr_energy_with_isdf
        mp._fswap = None
        polw_kpt = krpa_pol_with_isdf(mp, nw=nw)

        t0 = time.time()
        e_corr_krpa = krpa_corr_energy_with_isdf(mp, nw=nw, polw_kpt=polw_kpt)
        ene_krpa = e_corr_krpa + ene_krhf
        log.write("time_k_sos_lt%d_mp2 = % 6.2f\n" % (nw, time.time() - t0))
        log.write("ene_krpa_nw%d = % 12.8f\n" % (nw, ene_krpa))
        log.write("ene_corr_krpa_nw%d = % 12.8f\n" % (nw, e_corr_krpa))
        log.flush()

        s = "# nw = %3d, e_corr_krpa = %12.8f" % (nw, e_corr_krpa)
        print(s, flush=True)

if __name__ == "__main__":
    main()
