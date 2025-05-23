import numpy, scipy
import pyscf, fft
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

from pyscf.pbc.lno.tools import k2s_scf
from pyscf.data.elements import chemcore

def get_coeff_lo(kmf):
    from pyscf import lo
    from pyscf.pbc.lno.tools import k2s_aoint
    from pyscf.pbc.lno.tools import sort_orb_by_cell

    mf = k2s_scf(kmf)

    kpts = kmf.kpts
    nkpts = len(kpts)
    cell = kmf.cell
    scell = mf.cell

    frozen_per_cell = chemcore(cell)
    frozen = frozen_per_cell * nkpts

    # Localize occ mo in supercell (Unfortunately pyscf does not have a k-point PM)
    sorbocc = mf.mo_coeff[:,mf.mo_occ>1e-6][:,frozen:]
    s1e = k2s_aoint(cell, kpts, kmf.get_ovlp())

    from pyscf import lo
    mlo = lo.pipek.PipekMezey(scell, sorbocc)
    lo_coeff = mlo.kernel()
    while True: # Important: using jacobi sweep-based stability check to escape from local minimum
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(scell, lo_coeff1).set(verbose=4)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    lo_coeff_sorted = sort_orb_by_cell(scell, lo_coeff, nkpts, s=s1e)
    return lo_coeff_sorted

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
klno_ref.lno_type = ['2p', '2h']
klno_ref.verbose = 10
klno_ref.lo_proj_thresh_active = 1e-4
eris_ref = klno_ref.ao2mo()
# res = klno_ref.kernel()

klno_sol = KLNOCCSD(kmf_sol, coeff_lo_s, frag_lo_list, frozen=0, mf=smf)
klno_sol.lno_type = ['2p', '2h']
klno_sol.verbose = 10
klno_sol.lo_proj_thresh_active = 1e-4
eris_sol = klno_sol.ao2mo()

nfrag = len(frag_lo_list)
lno_type = klno_sol.lno_type
lno_thresh  = klno_sol.lno_thresh

for f, lo_ix_f in enumerate(frag_lo_list):
    coeff_lo_f = coeff_lo_s[:, lo_ix_f]
    param = [{"thresh": lno_thresh[x], "pct_occ": None, "norb": None} for x in range(2)]
    thresh_active = klno_ref.lo_proj_thresh_active
    
    s1e = klno_ref.s1e
    res = klno_ref.split_mo_coeff()
    orb_occ_frz_core = res[0]
    orb_occ = res[1]
    orb_vir = res[2]
    orb_vir_frz_core = res[3]
    e_occ, e_vir = klno_ref.split_mo_energy()[1:3]
    
    from functools import reduce
    from pyscf.lno.lno import projection_construction
    u_occ_loc = reduce(numpy.dot, (coeff_lo_f.T.conj(), s1e, orb_occ))
    u_occ_loc, u_occ_std, u_occ_orth = projection_construction(u_occ_loc, klno_ref.lo_proj_thresh, thresh_active)

    u_vir_loc = reduce(numpy.dot, (coeff_lo_f.T.conj(), s1e, orb_vir))
    u_vir_loc, u_vir_std, u_vir_orth = projection_construction(u_vir_loc, klno_ref.lo_proj_thresh, thresh_active)

    for lno_type in ['1h', '1p', '2p']:
        print(f"\n{lno_type = }")
        dm_oo_ref = klno_ref.make_lo_rdm1_occ(eris_ref, e_occ, e_vir, u_occ_loc, u_vir_loc, lno_type)
        dm_oo_ref = reduce(numpy.dot, (u_occ_orth.T.conj(), dm_oo_ref, u_occ_orth))

        dm_oo_sol = klno_sol.make_lo_rdm1_occ(eris_sol, e_occ, e_vir, u_occ_loc, u_vir_loc, lno_type)
        dm_oo_sol = reduce(numpy.dot, (u_occ_orth.T.conj(), dm_oo_sol, u_occ_orth))

        err = abs(dm_oo_ref - dm_oo_sol).max()
        print(f"{err = :6.4e}")

        print(f"{dm_oo_ref.shape = }")
        numpy.savetxt(cell.stdout, dm_oo_ref, fmt='% 6.4e', delimiter=', ')

        print(f"{dm_oo_sol.shape = }")
        numpy.savetxt(cell.stdout, dm_oo_sol, fmt='% 6.4e', delimiter=', ')
        assert err < 1e-4

    for lno_type in ['1h', '1p', '2h']:
        print(f"\n{lno_type = }")
        dm_vv_ref = klno_ref.make_lo_rdm1_vir(eris_ref, e_occ, e_vir, u_occ_loc, u_vir_loc, lno_type)
        dm_vv_ref = reduce(numpy.dot, (u_vir_orth.T.conj(), dm_vv_ref, u_vir_orth))

        dm_vv_sol = klno_sol.make_lo_rdm1_vir(eris_sol, e_occ, e_vir, u_occ_loc, u_vir_loc, lno_type)
        dm_vv_sol = reduce(numpy.dot, (u_vir_orth.T.conj(), dm_vv_sol, u_vir_orth))

        err = abs(dm_vv_ref - dm_vv_sol).max()
        print(f"{err = :6.4e}")

        print(f"{dm_vv_ref.shape = }")
        numpy.savetxt(cell.stdout, dm_vv_ref[:10, :10], fmt='% 6.4e', delimiter=', ')

        print(f"{dm_vv_sol.shape = }")
        numpy.savetxt(cell.stdout, dm_vv_sol[:10, :10], fmt='% 6.4e', delimiter=', ')
        assert err < 1e-4
