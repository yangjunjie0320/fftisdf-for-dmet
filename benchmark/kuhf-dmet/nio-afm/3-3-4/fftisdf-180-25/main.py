import os, sys, numpy, time
import pyscf, libdmet, fft

def main():
    from utils import build
    config = build()

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

    df_path = config["df_path"]
    if os.path.exists(df_path):
        df_size_in_byte = float(os.path.getsize(df_path))
        df_size_in_giga = df_size_in_byte / (1024 ** 3)
        print("df_size_in_byte = % 6.2e" % df_size_in_byte)
        print("df_size_in_giga = % 6.2e" % df_size_in_giga)

        log.write("df_size_in_gb = % 6.2e\n" % df_size_in_giga)

    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()
    if naux is not None:
        log.write("naux = %d\n" % naux)
    
    scf_obj = config["mf"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    
    t0 = time.time()
    vjk = scf_obj.get_jk(dm_kpts=dm0, hermi=1, with_j=True, with_k=False)
    log.write("time_get_j = % 6.2f\n" % (time.time() - t0))
    log.flush()

    t0 = time.time()
    vjk = scf_obj.get_jk(dm_kpts=dm0, hermi=1, with_j=False, with_k=True)
    log.write("time_get_k = % 6.2f\n" % (time.time() - t0))
    log.flush()

    t0 = time.time()
    ene_krhf = scf_obj.kernel(dm0)
    dm0 = scf_obj.make_rdm1()
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()
    
    import dmet
    from dmet import build_dmet
    from dmet import get_emb_eri_fftisdf
    get_emb_eri_old = libdmet.basis.trans_2e.get_emb_eri    

    def get_emb_eri(*args, **kwargs):
        df_obj = args[0]
        eri_emb = None

        t0 = time.time()
        if isinstance(df_obj, fft.ISDF):
            kwargs.pop('use_mpi')
            eri_emb = get_emb_eri_fftisdf(*args, **kwargs)
        else:
            eri_emb = get_emb_eri_old(*args, **kwargs)

        log.write("time_get_eri = % 6.2f\n" % (time.time() - t0))
        assert eri_emb is not None
        return eri_emb
    libdmet.basis.trans_2e.get_emb_eri = get_emb_eri
    
    build_dmet(config)

    emb = config["emb"]
    emb.max_cycle = 50
    emb.kernel()
    ene_dmet = emb.e_tot
    
    log.write("time_dmet = % 6.2f\n" % (time.time() - t0))
    log.write("ene_dmet = % 12.8f\n" % ene_dmet)
    log.flush()


if __name__ == "__main__":
    main()
