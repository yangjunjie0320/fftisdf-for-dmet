import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    table = {}
    df_obj = config["df"]
    t0 = time.time()
    df_obj.blksize = 20000
    df_obj.build()
    table["time_build_df"] = time.time() - t0

    scf_obj = config["mf"]
    dm0 = config["dm0"]
    scf_obj.with_df = df_obj
    ene_kscf = scf_obj.kernel(dm0)
    scf_obj.analyze()

    t0 = time.time()
    dm0 = scf_obj.make_rdm1()
    vj, vk = scf_obj.get_jk(dm_kpts=dm0, hermi=1, with_j=False, with_k=True)
    table["time_get_vk"] = time.time() - t0

    t0 = time.time()
    klno_obj = None
    if isinstance(df_obj, fft.ISDF):
        from klno import KLNOCCSD
        klno_obj = KLNOCCSD(scf_obj, thresh=1e-4)
    else:
        from lno.cc import KLNOCCSD
        klno_obj = KLNOCCSD(scf_obj, thresh=1e-4)

    klno_obj.no_type = "edmet"
    klno_obj.lo_type = "iao"
    klno_obj.verbose = 5
    klno_obj.kernel()
    table["time_klno"] = time.time() - t0

    ene_klno_mp2 = klno_obj.e_tot_pt2
    ene_klno_ccsd = klno_obj.e_tot

    nao = scf_obj.cell.nao_nr()
    nkpt = len(scf_obj.kpts)
    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()
    assert naux is not None

    with open("out.log", "w") as f:
        f.write("method = %s\n" % config["density_fitting_method"])
        f.write("basis = %s\n" % config["basis"])
        f.write("ke_cutoff = %6.2f\n" % config["ke_cutoff"])
        f.write("nao = %d\n" % nao)
        if naux is not None:
            f.write("naux = %d\n" % naux)
        f.write("nkpt = %d\n" % nkpt)
        f.write("ene_krhf = % 12.8f\n" % ene_kscf)
        f.write("ene_klno_mp2 = % 12.8f\n" % ene_klno_mp2)
        f.write("ene_klno_ccsd = % 12.8f\n" % ene_klno_ccsd)

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
    parser.add_argument("--ke-cutoff", type=float, required=True)
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
