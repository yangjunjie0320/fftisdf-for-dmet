import os, sys, numpy, time

import pyscf, fft
from pyscf import lib

def main(config: dict):
    from utils import build
    build(config)

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
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    t0 = time.time()
    from pyscf.pbc.mp import KMP2
    mp_obj = KMP2(scf_obj)
    mp_obj.verbose = 10

    from krpa import krpa_pol_with_isdf, krpa_corr_energy_with_isdf
    t0 = time.time()
    nw = 40
    polw_kpt = krpa_pol_with_isdf(mp_obj, nw=nw)
    log.write("time_polw_kpt = % 6.2f\n" % (time.time() - t0))
    log.flush()

    t0 = time.time()
    e_corr_krpa = krpa_corr_energy_with_isdf(mp_obj, nw=nw, polw_kpt=polw_kpt)
    log.write("time_krpa = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krpa = % 12.8f\n" % (e_corr_krpa + ene_krhf))
    log.write("corr_krpa = % 12.8f\n" % e_corr_krpa)
    log.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--density-fitting-method", type=str, required=True)
    parser.add_argument("--is-unrestricted", action="store_true")
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--kmesh", type=str, required=True)
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--df-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__

    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)

