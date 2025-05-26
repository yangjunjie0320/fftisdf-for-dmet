import numpy, scipy
import pyscf, fft
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

from pyscf.pbc.lno.tools import k2s_scf, k2s_iao

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
cell.verbose = 4
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

kmf_sol = khf.KRHF(cell, kpts, exxdiv=None)
kmf_sol.chkfile = 'kmf-scf.chk'
kmf_sol.init_guess = 'chkfile'
kmf_sol.conv_tol = tol

kmf_sol.with_df = fft.ISDF(cell, kpts)
kmf_sol.with_df.verbose = 0
kmf_sol.with_df._isdf = 'kmf-isdf.chk'
kmf_sol.with_df._isdf_to_save = 'kmf-isdf.chk'
kmf_sol.with_df.c0 = 10.0
kmf_sol.with_df.build()
kmf_sol.kernel()

kmf_ref = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()
kmf_ref.chkfile = 'kmf-scf.chk'
kmf_ref.init_guess = 'chkfile'
kmf_ref.with_df._cderi = 'kmf-rsdf.chk'
kmf_ref.with_df._cderi_to_save = 'kmf-rsdf.chk'
kmf_ref.kernel()

ene_krhf_sol = kmf_sol.e_tot
ene_krhf_ref = kmf_ref.e_tot
err_ene_krhf = abs(ene_krhf_sol - ene_krhf_ref)
assert err_ene_krhf < 1e-4
print(f"{ene_krhf_sol = :12.8f}, {ene_krhf_ref = :12.8f}, {err_ene_krhf = :6.4e}")

smf = k2s_scf(kmf_ref)
orb_occ_k = []
for k in range(nkpt):
    coeff_k = kmf_ref.mo_coeff[k]
    nocc_k = numpy.count_nonzero(kmf_ref.mo_occ[k])
    orb_occ_k.append(coeff_k[:, 0:nocc_k])

from pyscf.pbc.lno.tools import k2s_iao
coeff_lo_s = k2s_iao(cell, orb_occ_k, kpts, orth=True)

nlo_per_img = coeff_lo_s.shape[1] // nimg
frag_lo_list = [[f] for f in range(nlo_per_img)]

from klno import KLNOCCSD
klno_ref = KLNOCCSD(kmf_ref, coeff_lo_s, frag_lo_list, frozen=0, mf=smf)
klno_ref.lno_type = ['1h', '1h']
klno_ref.lno_thresh = [1e-4, 1e-5]
klno_ref.verbose = 10
klno_ref.kwargs_imp = {'max_cycle': 100, "verbose": 5}
klno_ref.lo_proj_thresh_active = 1e-4
res = klno_ref.kernel()

klno_sol = KLNOCCSD(kmf_sol, coeff_lo_s, frag_lo_list, frozen=0, mf=smf)
klno_sol.lno_type = ['1h', '1h']
klno_sol.lno_thresh = [1e-4, 1e-5]
klno_sol.verbose = 10
klno_sol.kwargs_imp = {'max_cycle': 100, "verbose": 5}
klno_sol.lo_proj_thresh_active = 1e-4
res = klno_sol.kernel()
