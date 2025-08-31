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

    t0 = time.time()
    from pyscf.pbc.lno.tools import k2s_scf
    mf_s = k2s_scf(scf_obj)
    orb_occ_k = []
    for k in range(nkpt):
        coeff_k = scf_obj.mo_coeff[k]
        nocc_k = numpy.count_nonzero(scf_obj.mo_occ[k])
        orb_occ_k.append(coeff_k[:, 0:nocc_k])

    from pyscf.pbc.lno.tools import k2s_iao
    coeff_lo_s = k2s_iao(scf_obj.cell, orb_occ_k, scf_obj.kpts, orth=True)
    coeff_lo_s = coeff_lo_s.real
    nlo = coeff_lo_s.shape[1] // nimg
    nlo_s = nlo * nimg
    assert coeff_lo_s.shape[1] == nlo_s
    frag_lo_list = [[f] for f in range(nlo)]
    print(f"nlo = {nlo}, nlo_s = {nlo_s}")
    print(f"coeff_lo_s.shape = {coeff_lo_s.shape}")
    print(f"frag_lo_list = {frag_lo_list}")

    from klno import KLNOCCSD
    klno_obj = KLNOCCSD(scf_obj, coeff_lo_s, frag_lo_list, frozen=0, mf=mf_s)
    klno_obj.lno_type = ["2p", "2h"]

    # local orbital 
    klno_obj.lo_proj_thresh_active = None

    # lno_thresh in [1e-4, 1e-6, 1e-8, 1e-9, 1e-10]
    lno_thresh = config["lno_thresh"]
    gamma = 10
    klno_obj.lno_thresh = [gamma * lno_thresh, lno_thresh]
    klno_obj.verbose = 5
    klno_obj.verbose_imp = 5
    klno_obj.kernel()
    log.write("time_klno = % 6.2f\n" % (time.time() - t0))

    ene_corr_klno_mp2 = klno_obj.e_corr_pt2
    ene_corr_klno_ccsd = klno_obj.e_corr
    ene_corr_pt2_os = klno_obj.e_corr_pt2_os
    ene_klno_mp2 = ene_corr_klno_mp2 + ene_krhf
    ene_klno_ccsd = ene_corr_klno_ccsd + ene_krhf

    log.write("ene_klno_mp2 = % 12.8f\n" % ene_klno_mp2)
    log.write("ene_klno_ccsd = % 12.8f\n" % ene_klno_ccsd)
    log.write("ene_klno_corr_mp2 = % 12.8f\n" % ene_corr_klno_mp2)
    log.write("ene_klno_corr_ccsd = % 12.8f\n" % ene_corr_klno_ccsd)
    log.write("ene_klno_corr_os = % 12.8f\n" % ene_corr_pt2_os)
    log.flush()


if __name__ == "__main__":
    main()
