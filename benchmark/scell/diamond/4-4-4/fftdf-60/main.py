import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    method = config["density_fitting_method"]
    assert "fftdf" in method, "only fftdf is supported for scell"
    ke_cutoff = float(method.split("-")[1])

    cell = config["cell"]
    cell.verbose = 5
    cell.ke_cutoff = ke_cutoff
    cell.build(dump_input=False)

    print("ke_cutoff = %6.2f, mesh = %s" % (ke_cutoff, cell.mesh))
    print("cell.atom_coords = ", cell.atom_coords)
    print("cell = \n", cell.tostring())

    cell = config["scell"]
    cell.verbose = 5
    cell.ke_cutoff = ke_cutoff
    cell.build(dump_input=False)

    print("ke_cutoff = %6.2f, mesh = %s" % (ke_cutoff, cell.mesh))
    print("cell.atom_coords = ", cell.atom_coords)
    print("cell = \n", cell.tostring())

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
    scf_obj.verbose = 5
    scf_obj.conv_tol = 1e-6
    ene_krks = scf_obj.kernel()
    log.write("time_krks = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krks = % 12.8f\n" % ene_krks)
    log.flush()

    t0 = time.time()
    scf_obj = config["mf"]
    scf_obj = pyscf.pbc.scf.RHF(cell)
    scf_obj.exxdiv = "ewald"
    scf_obj.verbose = 5
    scf_obj.conv_tol = 1e-6
    ene_krhf = scf_obj.kernel()
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    t0 = time.time()
    from pyscf.pbc.mp import MP2
    mp_obj = MP2(scf_obj)
    mp_obj.verbose = 10
    mp_obj.kernel(with_t2=False)
    log.write("time_kmp2 = % 6.2f\n" % (time.time() - t0))
    log.write("ene_kmp2 = % 12.8f\n" % mp_obj.e_tot)
    log.write("ene_corr_kmp2 = % 12.8f\n" % mp_obj.e_corr)
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

