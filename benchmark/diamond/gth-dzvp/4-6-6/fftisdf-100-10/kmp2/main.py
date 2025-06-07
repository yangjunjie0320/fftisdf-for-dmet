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
    from pyscf.pbc.mp import KMP2
    mp_obj = KMP2(scf_obj)
    mp_obj.verbose = 10
    mp_obj.kernel()
    table["time_kmp2"] = time.time() - t0

    ene_kmp2 = mp_obj.e_tot
    ene_corr_kmp2 = mp_obj.e_corr

    ene_kccsd = None
    ene_corr_kccsd = None
    try:
        from pyscf.pbc.cc import KCCSD
        cc_obj = KCCSD(scf_obj)
        cc_obj.verbose = 10
        eris = cc_obj.ao2mo()
        cc_obj.kernel(eris=eris)
        ene_kccsd = cc_obj.e_tot
        ene_corr_kccsd = cc_obj.e_corr
    except Exception as e:
        print(e)
        pass

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
        f.write("ene_kmp2 = % 12.8f\n" % ene_kmp2)
        f.write("ene_corr_kmp2 = % 12.8f\n" % ene_corr_kmp2)

        if ene_kccsd is not None:
            f.write("ene_kccsd = % 12.8f\n" % ene_kccsd)
            f.write("ene_corr_kccsd = % 12.8f\n" % ene_corr_kccsd)

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
