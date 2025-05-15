import numpy, scipy
import pyscf, fft, lno
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

THRESH_INTERNAL = 1e-10

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
cell.ke_cutoff = 120.0
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

kmf_sol = khf.KRHF(cell, kpts, exxdiv=None)
kmf_sol.chkfile = 'kmf-scf.chk'
kmf_sol.init_guess = 'chkfile'
kmf_sol.conv_tol = tol

kmf_sol.with_df = fft.ISDF(cell, kpts)
kmf_sol.with_df.verbose = 0
kmf_sol.with_df._isdf = 'kmf-isdf.chk'
kmf_sol.with_df._isdf_to_save = 'kmf-isdf.chk'
kmf_sol.with_df.c0 = 20.0
kmf_sol.with_df.build()
kmf_sol.kernel()

import lno.base.klno
kmf_ref = khf.KRHF(cell, kpts, exxdiv=None).rs_density_fit()
kmf_ref.chkfile = 'kmf-scf.chk'
kmf_ref.init_guess = 'chkfile'
kmf_ref.with_df._cderi_to_save = 'kmf-rsdf.chk'
kmf_ref.with_df._cderi = 'kmf-rsdf.chk'
kmf_ref.kernel()

ene_krhf_sol = kmf_sol.e_tot
ene_krhf_ref = kmf_ref.e_tot
err_ene_krhf = abs(ene_krhf_sol - ene_krhf_ref)
assert err_ene_krhf < 1e-4
print(f"{ene_krhf_sol = :12.8f}, {ene_krhf_ref = :12.8f}, {err_ene_krhf = :6.4e}")

import klno
klno_sol = klno.KLNOCCSD(kmf_sol, thresh=1e-4)
klno_sol.no_type = 'edmet'
klno_sol.lo_type = 'iao'
klno_sol.verbose = 5

import lno.cc
klno_ref = lno.cc.KLNOCCSD(kmf_ref, thresh=1e-4)
klno_ref.no_type = 'edmet'
klno_ref.lo_type = 'iao'
klno_ref.verbose = 5

smf = klno_sol._scf
kmf = klno_sol._kscf
mo_occ_s = smf.mo_occ

lo_type = klno_sol.lo_type
no_type = "rr" if klno_sol.no_type == "edmet" else None
assert lo_type == "iao"
assert no_type == "rr"

nocc = numpy.count_nonzero(mo_occ_s > 0)
frozen = klno_sol.frozen
if frozen is None:
    frozen = 0
coeff_occ = smf.mo_coeff[:, frozen:nocc]

from lno.base.lno import get_iao
coeff_lo = get_iao(smf.cell, coeff_occ, minao="minao", orth=True)
nlo = coeff_lo.shape[1]
nlo_per_img = nlo // nimg

frag_lo_list = [[f] for f in range(nlo_per_img)]
nfrag = len(frag_lo_list)

frag_aotm_list = None
frag_wght_list = numpy.ones(nfrag)
frag_nonv_list = [[None, None]] * nfrag

eris_ref = klno_ref.ao2mo()
eris_sol = klno_sol.ao2mo()

coeff_sol = klno_sol._scf.mo_coeff
coeff_ref = klno_ref._scf.mo_coeff

h1e_sol = klno_sol._scf.get_hcore()
h1e_ref = klno_ref._scf.get_hcore()
err_h1e = abs(h1e_sol - h1e_ref).max()
print(f"{err_h1e = :6.4e}")

ovlp_sol = klno_sol._scf.get_ovlp()
ovlp_ref = klno_ref._scf.get_ovlp()
err_ovlp = abs(ovlp_sol - ovlp_ref).max()
print(f"{err_ovlp = :6.4e}")

dm0 = klno_sol._scf.make_rdm1()
vhf_sol = klno_sol._scf.get_veff()
vhf_ref = klno_ref._scf.get_veff()
err_vhf = abs(vhf_sol - vhf_ref).max()

f1e_sol = klno_sol._scf.get_fock()
f1e_ref = klno_ref._scf.get_fock()
err_f1e = abs(f1e_sol - f1e_ref).max()
print(f"{err_f1e = :6.4e}")
assert 1 == 2

for f in range(nfrag):
    frag_lo_f = frag_lo_list[f]
    coeff_lo_f = coeff_lo[:, frag_lo_f]
    frag_target_nocc, frag_target_nvir = frag_nonv_list[f]
    assert frag_target_nocc is None
    assert frag_target_nvir is None

    frozen_mask = klno_sol.get_frozen_mask()
    thresh_pno = [klno_sol.thresh_occ, klno_sol.thresh_vir]
    
    print("\nRef")
    frozen_frag_ref, coeff_f_ref = klno_ref.make_fpno1(
        eris_ref, coeff_lo_f, no_type, THRESH_INTERNAL,
        thresh_pno, frozen_mask, None, None,
    )

    print("\nSol")
    frozen_frag_sol, coeff_f_sol = klno_sol.make_fpno1(
        eris_sol, coeff_lo_f, no_type, THRESH_INTERNAL,
        thresh_pno, frozen_mask, None, None,
    )

    res = klno_ref.impurity_solve(
        klno_ref._scf, coeff_f_ref, coeff_lo_f, eris_ref,
        frozen=frozen_frag_ref, log=log,
    )

    print(f"{res = }")

    res = klno_sol.impurity_solve(
        klno_sol._scf, coeff_f_sol, coeff_lo_f, eris_sol,
        frozen=frozen_frag_sol, log=log,
    )

    print(f"{res = }")

    assert 1 == 2

assert 1 == 2
klno_sol.kernel()
klno_ref.kernel()

# ene_krhf_sol = kmf_sol.e_tot
# ene_krhf_ref = kmf_ref.e_tot
# err_ene_krhf = abs(ene_krhf_sol - ene_krhf_ref)

# ene_klno_sol = klno_sol.e_tot
# ene_klno_ref = klno_ref.e_tot
# err_ene_klno = abs(ene_klno_sol - ene_klno_ref)

# print("ene_krhf_sol = %12.8f, ene_krhf_ref = %12.8f, err_ene_krhf = %12.8f" % (ene_krhf_sol, ene_krhf_ref, err_ene_krhf))
# print("ene_klno_sol = %12.8f, ene_klno_ref = %12.8f, err_ene_klno = %12.8f" % (ene_klno_sol, ene_klno_ref, err_ene_klno))
