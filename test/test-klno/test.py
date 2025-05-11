import numpy, scipy
import pyscf, fft, lno
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

cell = gto.Cell()
cell.atom = '''
C 0.0000 0.0000 0.0000
C 0.8917 0.8917 0.8917
'''
cell.a = '''
0.0000 1.7834 1.7834
1.7834 0.0000 1.7834
1.7834 1.7834 0.0000
'''
cell.unit = 'A'
cell.ke_cutoff = 100.0
cell.verbose = 0
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pbe'
cell.exp_to_discard = 0.1
cell.build(dump_input=False)

kmesh = [1, 1, 3]
kpts = cell.make_kpts(kmesh)

def setup_kmf_with_rsdf():
    kmf = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()
    kmf.verbose = 4

    kmf.with_df._cderi_to_save = 'kmf-rsgdf.chk'
    # kmf.with_df._cderi = 'kmf-rsgdf.chk'
    kmf.conv_tol = 1e-6
    kmf.verbose = 4
    kmf.chkfile = 'kmf-scf.chk'
    kmf.init_guess = 'chkfile'
    kmf.kernel()

    from lno.cc import KLNOCCSD
    lno_obj = KLNOCCSD(kmf, thresh=1e-4)
    eris_obj = lno_obj.ao2mo()
    lno_obj.no_type = 'edmet'
    lno_obj.kernel(eris=eris_obj)
    return kmf, lno_obj

def setup_kmf_with_isdf():
    kmf = khf.KRHF(cell, kpts, exxdiv=None)
    kmf.verbose = 4

    kmf.with_df = fft.ISDF(cell, kpts)
    kmf.with_df.c0 = 20.0
    kmf.with_df._isdf_to_save = 'kmf-isdf.chk'
    kmf.with_df._isdf = 'kmf-isdf.chk'
    kmf.with_df.build()

    kmf.conv_tol = 1e-6
    kmf.verbose = 4
    kmf.chkfile = 'kmf-scf.chk'
    kmf.init_guess = 'chkfile'
    kmf.kernel()

    import klno
    lno_obj = klno.KLNOCCSD(kmf, thresh=1e-4)
    eris_obj = lno_obj.ao2mo()
    lno_obj.no_type = 'edmet'
    lno_obj.kernel(eris=eris_obj)
    return kmf, lno_obj

if __name__ == '__main__':
    kmf_sol, lno_obj_sol = setup_kmf_with_isdf()
    kmf_ref, lno_obj_ref = setup_kmf_with_rsdf()
    print("ene_kmf_isdf = %12.8f, ene_ccsd_isdf = %12.8f" % (kmf_sol.e_tot, lno_obj_sol.e_tot))
    print("ene_kmf_rsdf = %12.8f, ene_ccsd_rsdf = %12.8f" % (kmf_ref.e_tot, lno_obj_ref.e_tot))
