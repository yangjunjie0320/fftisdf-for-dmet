import numpy, scipy, os
import pyscf
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

from pyscf.pbc.lno.tools import k2s_scf, k2s_iao
from pyscf.data.elements import chemcore


def get_coeff_lo(kmf):
    cell = kmf.cell
    smf = k2s_scf(kmf)
    scell = smf.cell
    nkpt = nimg = len(kmf.kpts)

    frozen_per_img = chemcore(cell)
    frozen = frozen_per_img * nimg

    mo_coeff_s = smf.mo_coeff
    mo_occ_s  = smf.mo_occ
    orb_occ_s = mo_coeff_s[:, mo_occ_s>1e-6]
    orb_occ_s = orb_occ_s[:, frozen:]

    from pyscf import lo
    lo_obj = lo.pipek.PipekMezey(scell, orb_occ_s)
    coeff_lo_prev = lo_obj.kernel()
    coeff_lo_next = None

    while True:
        coeff_lo_next = lo_obj.stability_jacobi()[1]
        if coeff_lo_next is coeff_lo_prev:
            break
        lo_obj = lo.PipekMezey(scell, coeff_lo_next).set(verbose=4)
        lo_obj.init_guess = None
        coeff_lo_prev = lo_obj.kernel()

    assert coeff_lo_next is not None
    
    from pyscf.pbc.lno.tools import sort_orb_by_cell
    from pyscf.pbc.lno.tools import k2s_aoint
    s1e = k2s_aoint(cell, kmf.kpts, kmf.get_ovlp())
    coeff_lo_s = sort_orb_by_cell(scell, coeff_lo_next, nimg, s=s1e)
    return coeff_lo_s

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

kmf_obj = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()
kmf_obj.chkfile = 'kmf-scf.chk'
kmf_obj.with_df._cderi_to_save = 'kmf-rsdf.chk'

if os.path.exists('kmf-scf.chk'):
    kmf_obj.init_guess = 'chkfile'

if os.path.exists('kmf-rsdf.chk'):
    kmf_obj.with_df._cderi = 'kmf-rsdf.chk'

kmf_obj.kernel()

coeff_lo_s = get_coeff_lo(kmf_obj)
nlo_per_img = coeff_lo_s.shape[1] // nimg
frag_lo_list = [[f] for f in range(nlo_per_img)]

frozen_per_img = chemcore(cell)
frozen = frozen_per_img * nimg

from pyscf.pbc.lno.klnoccsd import KLNOCCSD
klno_ref = KLNOCCSD(kmf_obj, coeff_lo_s, frag_lo_list, frozen=0, mf=None)
klno_ref.lno_type = ['1h', '1h']
klno_ref.lno_thresh = [1e-4, 1e-5]
klno_ref.verbose = 10
klno_ref.kwargs_imp = {'max_cycle': 100, "verbose": 5}
res = klno_ref.kernel()
