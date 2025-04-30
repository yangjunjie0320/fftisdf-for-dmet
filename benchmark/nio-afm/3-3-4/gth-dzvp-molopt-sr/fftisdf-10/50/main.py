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
    vj, vk = scf_obj.get_jk(dm_kpts=dm0, hermi=1, with_j=False, with_k=True)
    table["time_get_vk"] = time.time() - t0

    latt = config["lattice"]
    is_unrestricted = config["is_unrestricted"]

    from dmet import build_dmet
    from dmet import get_emb_eri_fftisdf_v1 as get_emb_eri_fftisdf_ref
    from dmet import get_emb_eri_fftisdf_v2 as get_emb_eri_fftisdf_sol
    get_emb_eri_old = libdmet.basis.trans_2e.get_emb_eri
    def get_emb_eri(*args, **kwargs):
        df_obj = args[0]
        eri_emb = None

        t0 = time.time()
        if isinstance(df_obj, fft.ISDF):
            kwargs.pop('use_mpi')
            t0 = time.time()
            eri_emb_sol = get_emb_eri_fftisdf_sol(*args, **kwargs)
            table["time_get_eri_sol"] = time.time() - t0
            t0 = time.time()
            eri_emb_ref = get_emb_eri_fftisdf_ref(*args, **kwargs)
            table["time_get_eri_ref"] = time.time() - t0

            for ss in range(3):
                err = abs(eri_emb_sol[ss] - eri_emb_ref[ss]).max()
                print("\nss = %d, error = %6.2e" % (ss, err))

            err = abs(eri_emb_sol - eri_emb_ref).max()
            eri_emb = eri_emb_sol
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

    naux = None
    from pyscf.pbc.df import GDF
    if isinstance(df_obj, GDF):
        naux = df_obj.get_naoaux()
    elif isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]

    with open("out.log", "w") as f:
        f.write("method = %s\n" % config["density_fitting_method"])
        f.write("basis = %s\n" % config["basis"])
        f.write("ke_cutoff = %6.2f\n" % config["ke_cutoff"])
        f.write("nao = %d\n" % nao)
        if naux is not None:
            f.write("naux = %d\n" % naux)
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
    parser.add_argument("--is-unrestricted", action="store_true")
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--df-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)
