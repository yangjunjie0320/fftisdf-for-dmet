import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    cell = config["cell"]
    kmesh = config["kmesh"]
    from pyscf.pbc.tools.k2gamma import get_phase
    scell, phase = get_phase(cell, kmesh)
    cell = scell

    natm = cell.natm
    nkpt, nimg = phase.shape

    log = open("out.log", "w")
    log.write("method = %s\n" % config["density_fitting_method"])
    log.write("basis = %s\n" % config["basis"])
    log.write("natm = %d\n" % natm)
    log.write("nimg = %d\n" % nimg)
    log.flush()

    t0 = time.time()
    if 'gdf' in config["density_fitting_method"]:
        print("Using GDF")
        method = config["density_fitting_method"].split("-")

        df_obj = pyscf.pbc.df.GDF(cell)
        df_obj.exxdiv = None

        if len(method) == 2:
            beta = float(method[1])
            print(f"Using beta = {beta}")
            from pyscf.df import aug_etb
            df_obj.auxbasis = aug_etb(cell, beta=beta)
        else:
            print("Using default settings for GDF")

    else:
        print("Using FFTDF")
        method = config["density_fitting_method"].split("-")
        assert len(method) == 2, f"Invalid method: {method}"

        from pyscf.pbc.df import FFTDF
        cell.ke_cutoff = float(method[1])
        cell.build(dump_input=False)

        df_obj = FFTDF(cell)
        df_obj.exxdiv = None
        print(f"ke_cutoff = {cell.ke_cutoff}, mesh = {df_obj.mesh}")

    log.write("time_build_df = % 6.2f\n" % (time.time() - t0))

    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()
    if naux is not None:
        log.write("naux = %d\n" % naux)

    t0 = time.time()
    scf_obj = pyscf.pbc.dft.RKS(cell)
    scf_obj.xc = "pbe"
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krks = scf_obj.kernel()
    log.write("time_krks = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krks = % 12.8f\n" % ene_krks)
    log.flush()

    t0 = time.time()
    scf_obj = pyscf.pbc.scf.RHF(cell)
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krhf = scf_obj.kernel()
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
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
