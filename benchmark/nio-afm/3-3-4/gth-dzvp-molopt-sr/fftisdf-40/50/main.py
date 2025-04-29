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
    scf_obj.with_df = df_obj
    ene_kscf = scf_obj.kernel(dm0)
    scf_obj.analyze()

    t0 = time.time()
    dm0 = scf_obj.make_rdm1()
    vj, vk = scf_obj.get_jk(dm_kpts=dm0)
    table["time_vjk"] = time.time() - t0

    latt = config["lattice"]
    is_unrestricted = config["is_unrestricted"]

    from trans2e import build_dmet, get_emb_eri_fftisdf
    get_emb_eri_old = libdmet.basis.trans_2e.get_emb_eri
    def get_emb_eri(*args, **kwargs):
        df_obj = args[0]
        eri_emb = None

        t0 = time.time()
        if isinstance(df_obj, fft.ISDF):
            kwargs.pop('use_mpi')
            from trans2e import get_emb_eri_fftisdf
            eri_emb = get_emb_eri_fftisdf(*args, **kwargs)
        else:
            eri_emb = get_emb_eri_old(*args, **kwargs)
        table["time_get_eri"] = time.time() - t0
        return eri_emb
    libdmet.basis.trans_2e.get_emb_eri = get_emb_eri

    t0 = time.time()
    emb_obj = build_dmet(scf_obj, latt, is_unrestricted)
    emb_obj.kernel()
    ene_dmet = emb_obj.e_tot
    nao = scf_obj.cell.nao_nr()
    nkpt = len(scf_obj.kpts)

    with open("out.log", "w") as f:
        f.write("method = %s\n" % config["density_fitting_method"])
        f.write("basis = %s\n" % config["basis"])
        f.write("ke_cutoff = %s\n" % config["ke_cutoff"])
        f.write("nao = %d\n" % nao)
        f.write("nkpt = %d\n" % nkpt)
        f.write("ene_krhf = % 12.8f\n" % ene_kscf)
        f.write("ene_dmet = % 12.8f\n" % ene_dmet)

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
    parser.add_argument("--is-unrestricted", type=bool, default=False)
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--df-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)
