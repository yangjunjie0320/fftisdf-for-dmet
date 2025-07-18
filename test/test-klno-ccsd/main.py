import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    table = {}
    df_obj = config["df"]
    t0 = time.time()
    df_obj.build()
    table["time_build_df"] = time.time() - t0

    scf_obj = config["mf"]
    dm0 = config["dm0"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_kscf = scf_obj.kernel(dm0)
    nao = scf_obj.cell.nao_nr()
    scf_obj.analyze()

    kpts = scf_obj.kpts
    nkpt = nimg = len(kpts)

    t0 = time.time()
    dm0 = scf_obj.make_rdm1()
    vj, vk = scf_obj.get_jk(dm_kpts=dm0, hermi=1, with_j=False, with_k=True)
    table["time_get_vk"] = time.time() - t0

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
    klno_obj.lo_proj_thresh_active = 1e-2

    # lno_thresh in [1e-4, 1e-6, 1e-8, 1e-9, 1e-10]
    lno_thresh = config["lno_thresh"]
    klno_obj.lno_thresh = [10 * lno_thresh, lno_thresh]
    klno_obj.verbose = 5
    klno_obj.verbose_imp = 5
    klno_obj.kernel()
    table["time_klno"] = time.time() - t0

    # ene_klno_mp2 = klno_obj.e_tot_pt2 / nimg
    # ene_klno_ccsd = klno_obj.e_tot / nimg
    ene_corr_klno_mp2 = klno_obj.e_corr_pt2
    ene_corr_klno_ccsd = klno_obj.e_corr
    ene_klno_mp2 = ene_corr_klno_mp2 + ene_kscf
    ene_klno_ccsd = ene_corr_klno_ccsd + ene_kscf
    
    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()

    with open("out.log", "w") as f:
        f.write("method = %s\n" % config["density_fitting_method"])
        f.write("basis = %s\n" % config["basis"])
        f.write("nao = %d\n" % nao)
        f.write("natm = %d\n" % scf_obj.cell.natm)
        if naux is not None:
            f.write("naux = %d\n" % naux)
        f.write("nkpt = %d\n" % nkpt)
        f.write("kmesh = %s\n" % config["kmesh"])
        f.write("ene_krhf = % 12.8f\n" % ene_kscf)
        f.write("ene_klno_mp2 = % 12.8f\n" % ene_klno_mp2)
        f.write("ene_klno_ccsd = % 12.8f\n" % ene_klno_ccsd)
        f.write("ene_corr_klno_mp2 = % 12.8f\n" % ene_corr_klno_mp2)
        f.write("ene_corr_klno_ccsd = % 12.8f\n" % ene_corr_klno_ccsd)

        for k, v in table.items():
            f.write("%s = % 6.2f\n" % (k, max(v, 0.01)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--kmesh", type=str, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--pseudo", type=str, required=True)
    parser.add_argument("--lno-thresh", type=float, default=3e-5)
    parser.add_argument("--density-fitting-method", type=str, required=True)
    parser.add_argument("--is-unrestricted", action="store_true")
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--df-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)
