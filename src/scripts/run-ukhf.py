import os, sys, numpy, time

import pyscf, libdmet

import fft

TMPDIR = os.environ.get("PYSCF_TMPDIR", None)
assert os.path.exists(TMPDIR)

MAX_MEMORY = os.environ.get("PYSCF_MAX_MEMORY", 2000)
MAX_MEMORY = int(MAX_MEMORY)
assert MAX_MEMORY > 0

table = {}

get_emb_eri_old = libdmet.basis.trans_2e.get_emb_eri
def get_emb_eri(*args, **kwargs):
    df_obj = args[0]
    eri_emb = None
    if isinstance(df_obj, fft.ISDF):
        kwargs.pop('use_mpi')
        from trans2e import get_emb_eri_fftisdf
        eri_emb = get_emb_eri_fftisdf(*args, **kwargs)
    else:
        eri_emb = get_emb_eri_old(*args, **kwargs)
    return eri_emb
libdmet.basis.trans_2e.get_emb_eri = get_emb_eri

def main():
    from utils import parse, build
    config = parse()
    build(config)

    df_obj = config["df"]
    t0 = time.time()
    df_obj.build()
    table["time_build_df"] = time.time() - t0

    scf_obj = config["mf"]
    dm0 = config["dm0"]
    scf_obj.with_df = df_obj
    ene_kscf = scf_obj.kernel(dm0)

    t0 = time.time()
    dm0 = scf_obj.make_rdm1()
    vj, vk = scf_obj.get_jk(dm_kpts=dm0)
    table["time_vjk"] = time.time() - t0

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
    emb = config["emb"]
    emb.kernel()
    table["time_emb_kernel"] = time.time() - t0

    ene_dmet = emb.e_tot
    nao = scf_obj.cell.nao_nr()
    nkpt = len(emb.latt.kpts)

    with open("out.log", "w") as f:
        f.write("method = %s\n" % config["method"])
        f.write("basis = %s\n" % config["basis"])
        f.write("ke_cutoff = %s\n" % config["ke_cutoff"])
        f.write("nao = %d\n" % nao)
        f.write("nkpt = %d\n" % nkpt)
        f.write("ene_krhf = % 12.8f\n" % ene_kscf)
        f.write("ene_dmet = % 12.8f\n" % ene_dmet)

        for k, v in table.items():
            f.write("%s = % 6.2f\n" % (k, max(v, 0.01))
    
    
    

if __name__ == "__main__":
    main()
