import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    cell = config["cell"]
    df_obj = config["df"]
    kpts = config["kpts"]
    nkpt = nimg = len(kpts)

    dm0 = config["dm0"]
    nao = dm0.shape[0]
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
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    if config["density_fitting_method"] == "fftisdf":
        # run sos-kmp2 and krpa
        from krpa import krpa_pol_with_isdf, krpa_corr_energy_with_isdf
        from krpa import kmp2_corr_energy_with_isdf

        for nw in [20, 25, 30, 35, 40]:
            kmp2 = pyscf.pbc.mp.KMP2(scf_obj)
            kmp2.verbose = 5

            polw_kpt = krpa_pol_with_isdf(kmp2, nw=nw)

            t0 = time.time()
            e_corr_krpa = krpa_corr_energy_with_isdf(kmp2, nw=nw, polw_kpt=polw_kpt)
            ene_krpa = e_corr_krpa + ene_krhf
            log.write("time_krpa_nw%d = % 6.2f\n" % (nw, time.time() - t0))
            log.write("ene_krpa_nw%d = % 12.8f\n" % (nw, ene_krpa))
            log.write("ene_corr_krpa_nw%d = % 12.8f\n" % (nw, e_corr_krpa))
            log.flush()

            t0 = time.time()
            e_os = kmp2_corr_energy_with_isdf(kmp2, nw=nw, sos_factor=1.0)
            e_corr_kmp2 = e_os * 1.3
            ene_kmp2 = e_corr_kmp2 + ene_krhf
            log.write("time_kmp2_nw%d = % 6.2f\n" % (nw, time.time() - t0))
            log.write("ene_kmp2_nw%d = % 12.8f\n" % (nw, ene_kmp2))
            log.write("ene_corr_kmp2_nw%d = % 12.8f\n" % (nw, e_corr_kmp2))
            log.write("ene_os_kmp2_nw%d = % 12.8f\n" % (nw, e_os))
            log.flush()

    elif config["density_fitting_method"] == "gdf":
        from fcdmft.rpa.pbc.krpa import KRPA

        for nw in [20, 25, 30, 35, 40]:
            t0 = time.time()
            krpa = KRPA(scf_obj)
            krpa.verbose = 5
            krpa.kernel(nw=nw)
            log.write("time_krpa_nw%d = % 6.2f\n" % (nw, time.time() - t0))
            ene_krpa = krpa.e_tot
            e_corr_krpa = krpa.e_corr
            log.write("ene_krpa_nw%d = % 12.8f\n" % (nw, ene_krpa))
            log.write("ene_corr_krpa_nw%d = % 12.8f\n" % (nw, e_corr_krpa))
            log.flush()

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
