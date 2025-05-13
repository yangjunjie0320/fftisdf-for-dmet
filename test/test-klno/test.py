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
cell.ke_cutoff = 200.0
cell.verbose = 0
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pbe'
cell.exp_to_discard = 0.1
cell.max_memory = 1000
cell.build(dump_input=False)

kmesh = [1, 1, 3]
kpts = cell.make_kpts(kmesh)
nkpt = nimg = len(kpts)
tol = 1e-8

from pyscf.lib import logger
log = logger.new_logger(cell.stdout, 5)

def setup_kmf_with_rsdf():
    kmf = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()
    kmf.with_df.verbose = 5
    
    kmf.conv_tol = tol
    kmf.verbose = 4
    kmf.chkfile = 'kmf-scf.chk'
    kmf.init_guess = 'chkfile'
    kmf.kernel()

    from lno.cc import KLNOCCSD
    lno_obj = KLNOCCSD(kmf, thresh=1e-4)
    lno_obj.no_type = 'edmet'
    lno_obj.lo_type = 'iao' 
    lno_obj.kernel()
    return kmf, lno_obj

def setup_kmf_with_isdf():
    kmf = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()

    from fft import isdf_ao2mo
    kmf.with_df = fft.ISDF(cell, kpts)
    kmf.with_df.verbose = 5
    kmf.with_df.c0 = 20.0
    kmf.with_df.build()

    kmf.conv_tol = tol
    kmf.verbose = 4
    kmf.chkfile = 'kmf-scf.chk'
    # kmf.init_guess = 'chkfile'
    kmf.kernel()

    import klno
    lno_obj = klno.KLNOCCSD(kmf, thresh=1e-4)
    lno_obj.no_type = 'edmet'
    lno_obj.lo_type = 'iao'
    lno_obj.verbose = 5
    lno_obj.force_outcore_ao2mo = False
    lno_obj.kernel()
    return kmf, lno_obj

if __name__ == '__main__':
    kmf_sol, lno_obj_sol = setup_kmf_with_isdf()
    kmf_ref, lno_obj_ref = setup_kmf_with_rsdf()
    
    ene_hf_ref = kmf_ref.e_tot
    ene_hf_sol = kmf_sol.e_tot
    err_ene_hf = abs(ene_hf_ref - ene_hf_sol)
    print("ene_hf_ref = %12.8f, ene_hf_sol = %12.8f, err = %6.4e" % (ene_hf_ref, ene_hf_sol, err_ene_hf))

    ene_cc_ref = lno_obj_ref.e_tot
    ene_cc_sol = lno_obj_sol.e_tot
    err_ene_cc = abs(ene_cc_ref - ene_cc_sol)
    print("ene_cc_ref = %12.8f, ene_cc_sol = %12.8f, err = %6.4e" % (ene_cc_ref, ene_cc_sol, err_ene_cc))
