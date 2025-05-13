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
    kmf.with_df._cderi_to_save = 'kmf-rsgdf.chk'
    kmf.with_df._cderi = 'kmf-rsgdf.chk'
    
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
    kmf.with_df._cderi_to_save = 'kmf-rsgdf.chk'
    kmf.with_df._cderi = 'kmf-rsgdf.chk'
    rsdf_obj = kmf.with_df

    from fft import isdf_ao2mo
    kmf.with_df = fft.ISDF(cell, kpts)
    kmf.with_df.verbose = 5
    kmf.with_df._isdf = 'kmf-isdf.chk'
    kmf.with_df._isdf_to_save = 'kmf-isdf.chk'
    kmf.with_df.build()
    isdf_obj = kmf.with_df

    kmf.conv_tol = tol
    kmf.verbose = 4
    kmf.chkfile = 'kmf-scf.chk'
    kmf.init_guess = 'chkfile'
    kmf.kernel()

    import klno_backup
    from klno_backup import KLNOCCSD
    from lno.cc.kccsd import get_maskact

    lno_obj = KLNOCCSD(kmf, thresh=1e-4)
    lno_obj.no_type = 'edmet'
    lno_obj.lo_type = 'iao'
    lno_obj.verbose = 5
    lno_obj.force_outcore_ao2mo = False
    lno_obj.kernel()
    return kmf, lno_obj
    # mf = lno_obj._scf
    # mo_occ = mf.mo_occ
    # kmf = lno_obj._kscf

    # no_type = lno_obj.no_type
    # if no_type == 'edmet':
    #     no_type = 'rr'
    # else:
    #     raise NotImplementedError
    # assert lno_obj.lo_type == 'iao'
    
    # nocc = numpy.count_nonzero(mf.mo_occ > 1e-10)
    # frozen = lno_obj.frozen if lno_obj.frozen is not None else 0
    # orb_occ = mf.mo_coeff[:, frozen:nocc]

    # from lno.base.lno import get_iao
    # orb_loc = get_iao(mf.cell, orb_occ, minao='minao', orth=True)
    # nlo = orb_loc.shape[1]
    # nlo_per_img = nlo // nimg

    # frag_lo_list = [[f] for f in range(nlo_per_img)]

    # nfrag = len(frag_lo_list)
    # frag_atm_list = None
    # frag_wght_list = numpy.ones(nfrag)
    # frag_nonv_list = [[None, None]] * nfrag

    # lno_obj._kscf.with_df = rsdf_obj
    # eris_ref = lno.cc.KLNOCCSD.ao2mo(lno_obj)

    # lno_obj._kscf.with_df = isdf_obj
    # eris_sol = lno_obj.ao2mo()

    # for ifrag in range(nfrag):
    #     frag_lo = frag_lo_list[ifrag]
    #     orb_frag_loc = orb_loc[:, frag_lo]
    #     frag_target_nocc, frag_target_nvir = frag_nonv_list[ifrag]

    #     frozen_mask = lno_obj.get_frozen_mask()
    #     thresh_pno = [lno_obj.thresh_occ, lno_obj.thresh_vir]

    #     assert frag_target_nocc is None
    #     assert frag_target_nvir is None

    #     THRESH_INTERNAL = 1e-10

    #     lno_obj._kscf.with_df = rsdf_obj
    #     frozen_frag_ref, orb_frag_ref = lno_obj.make_fpno1(
    #         eris_ref, orb_frag_loc, no_type,
    #         THRESH_INTERNAL, thresh_pno,
    #         frozen_mask=frozen_mask,
    #         frag_target_nocc=frag_target_nocc,
    #         frag_target_nvir=frag_target_nvir
    #     )

    #     lno_obj._kscf.with_df = isdf_obj
    #     frozen_frag_sol, orb_frag_sol = lno_obj.make_fpno1(
    #         eris_sol, orb_frag_loc, no_type,
    #         THRESH_INTERNAL, thresh_pno,
    #         frozen_mask=frozen_mask,
    #         )

    #     frozen_frag = frozen_frag_ref

    #     frozen, mask = get_maskact(frozen_frag, mo_occ.size)
    #     orb_frag = orb_frag_ref
        
    #     lno_obj._kscf.with_df = rsdf_obj
    #     cc_ref = lno.cc.kccsd.K2GCCSD(
    #         lno_obj._kscf, mf=mf,
    #         frozen=frozen_frag,
    #         mo_coeff=orb_frag,
    #         mo_occ=mo_occ,
    #         )

    #     cc_ref._s1e = eris_ref.s1e
    #     cc_ref._h1e = eris_ref.h1e
    #     cc_ref._vhf = eris_ref.vhf
    #     eris_frag_ref = cc_ref.ao2mo()

    #     # try:
    #     #     from lno.cc.ccsd import impurity_solve
    #     #     res = impurity_solve(
    #     #         cc_ref, orb_frag, orb_frag_loc,
    #     #         mo_occ, mask, eris_ref,
    #     #         log=log, ccsd_t=False, verbose_imp=None
    #     #     )

    #     #     print(res[0])
    #     #     ene_mp2_ref = res[1][0]
    #     #     ene_cc_ref = res[1][1]
    #     # except:
    #     #     pass

    #     lno_obj._kscf.with_df = isdf_obj
    #     cc_sol = klno_backup.K2GCCSD(
    #         lno_obj._kscf, mf=mf,
    #         frozen=frozen_frag,
    #         mo_coeff=orb_frag,
    #         mo_occ=mo_occ,
    #         )
        
    #     cc_sol._s1e = eris_sol.s1e
    #     cc_sol._h1e = eris_sol.h1e
    #     cc_sol._vhf = eris_sol.vhf
    #     eris_frag_sol = cc_sol.ao2mo()
    #     # print("eris_frag_sol = ", eris_frag_sol)
    #     # for k, v in eris_frag_sol.__dict__.items():
    #     #     print(k, v)
    #     # assert 1 == 2
    #     t1_sol, t2_sol = cc_sol.init_amps(eris=eris_frag_sol)[1:]

    #     eris_ovov_ref = eris_frag_ref.ovov[()]
    #     eris_ovov_sol = eris_frag_sol.ovov[()]

    #     nocc, nvir = t1_sol.shape
    #     eris_ovov_ref = eris_ovov_ref.reshape(nocc * nvir, nocc * nvir)
    #     eris_ovov_sol = eris_ovov_sol.reshape(nocc * nvir, nocc * nvir)

    #     print("eris_ovov_ref = ")
    #     numpy.savetxt(cell.stdout, eris_ovov_ref[:10, :10], delimiter=', ', fmt='% 6.4f')
    #     print("eris_ovov_sol = ")
    #     numpy.savetxt(cell.stdout, eris_ovov_sol[:10, :10], delimiter=', ', fmt='% 6.4f')

    #     assert numpy.allclose(eris_frag_ref.mo_coeff, eris_frag_sol.mo_coeff)
    #     print(eris_frag_ref.mo_coeff.shape, eris_frag_ref.mo_coeff.dtype)
    #     print(eris_frag_sol.mo_coeff.shape, eris_frag_sol.mo_coeff.dtype)
    #     assert 1 == 2

    #     print("t1_sol = ", t1_sol)
    #     print("t1_ref = ", t1_ref)

    #     print("t2_sol = ", t2_sol)
    #     print("t2_ref = ", t2_ref)
    #     assert 1 == 2

    #     # try:
    #     #     res = impurity_solve(
    #     #         cc_sol, orb_frag, orb_frag_loc,
    #     #         mo_occ, mask, eris_sol,
    #     #         log=log, ccsd_t=False, verbose_imp=None
    #     #     )
    #     #     ene_mp2_sol = res[1][0]
    #     #     ene_cc_sol = res[1][1]
    #     #     print(res[0])
    #     # except:
    #     #     pass

    #     # print("ene_mp2_ref = %12.8f, ene_mp2_sol = %12.8f, err = %6.4e" % (ene_mp2_ref, ene_mp2_sol, abs(ene_mp2_ref - ene_mp2_sol)))
    #     # print("ene_cc_ref = %12.8f, ene_cc_sol = %12.8f, err = %6.4e" % (ene_cc_ref, ene_cc_sol, abs(ene_cc_ref - ene_cc_sol)))
    #     # assert 1 == 2

    #     # from lno.base.lno import kernel_1frag

    #     # # lno_obj._kscf.with_df = rsdf_obj
    #     # # res_ref = kernel_1frag(
    #     # #     lno_obj, eris_ref, orb_frag_loc, no_type,
    #     # #     frag_target_nocc=frag_target_nocc,
    #     # #     frag_target_nvir=frag_target_nvir
    #     # #     )

    #     # lno_obj._kscf.with_df = isdf_obj
    #     # res_sol = kernel_1frag(
    #     #     lno_obj, eris_sol, orb_frag_loc, no_type,
    #     #     frag_target_nocc=frag_target_nocc,
    #     #     frag_target_nvir=frag_target_nvir
    #     #     )
    

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
