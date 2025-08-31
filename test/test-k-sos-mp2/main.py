import os, sys, numpy, time

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

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

    naux = None
    if isinstance(df_obj, fft.ISDF):
        naux = df_obj.inpv_kpt.shape[1]
    else:
        naux = df_obj.get_naoaux()
    if naux is not None:
        log.write("naux = %d\n" % naux)
    log.flush()

    t0 = time.time()
    scf_obj = config["mf"]
    scf_obj.exxdiv = "ewald"
    scf_obj.with_df = df_obj
    ene_krhf = scf_obj.kernel(dm0)
    log.write("time_krhf = % 6.2f\n" % (time.time() - t0))
    log.write("ene_krhf = % 12.8f\n" % ene_krhf)
    log.flush()

    mp = pyscf.pbc.mp.KMP2(scf_obj)
    mp.verbose = 10
    
    e_os_ref = numpy.nan
    if os.path.exists("e-os-ref.txt"):
        with open("e-os-ref.txt", "r") as f:
            lines = f.readlines()
            print("lines = %s" % lines)
            e_os_ref = float(lines[0].split()[-1])
    print("e_os_ref = %12.8f" % e_os_ref)

    for nw in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        from krpa import kmp2_corr_energy_with_isdf
        e_os = kmp2_corr_energy_with_isdf(mp, nw=nw, sos_factor=1.0)
        e_corr = e_os * 1.3
        ene_kmp2 = e_corr + ene_krhf
        log.write("time_k_sos_lt%d_mp2 = % 6.2f\n" % (nw, time.time() - t0))
        log.write("ene_k_sos_lt%d_mp2 = % 12.8f\n" % (nw, ene_kmp2))
        log.write("ene_corr_k_sos_lt%d_mp2 = % 12.8f\n" % (nw, e_corr))
        log.write("ene_os_lt%d = % 12.8f\n" % (nw, e_os))
        log.flush()

        s = "nw = %3d, e_os = %12.8f" % (nw, e_os)
        if not numpy.isnan(e_os_ref):
            s += ", err = %6.4e" % (abs(e_os - e_os_ref))
        print(s)

if __name__ == "__main__":
    main()
