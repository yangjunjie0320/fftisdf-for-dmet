import numpy, scipy
import pyscf, fft

import lno.tools.k2gamma
def k2gamma(kmf, tol_fock_imag=1e-4):
    from pyscf.pbc import scf
    from pyscf.lib import logger
    
    log = logger.new_logger(kmf)
    cell = kmf.cell
    kpts = kmf.kpts
    nkpt = len(kpts)
    nao = cell.nao_nr()

    from lno.tools.k2gamma import get_k2g_phase
    scell,  phase, kmesh = get_k2g_phase(cell, kpts)
    log.info('K2Gamma: found kmesh= %s', kmesh)
    nimg = phase.shape[1]
    assert nimg == nkpt

    e_mo_k = numpy.asarray(kmf.mo_energy)
    c_mo_k = numpy.asarray(kmf.mo_coeff)
    nmo = c_mo_k.shape[-1]

    e_mo_g = numpy.hstack(e_mo_k)
    c_mo_g = numpy.einsum('kw,kum->wukm', phase.conj(), c_mo_k)
    c_mo_g = c_mo_g.reshape(nao * nimg, nmo * nimg)

    n_mo_g = numpy.hstack(kmf.mo_occ)
    assert n_mo_g.shape == (nmo * nimg,)
    mask = n_mo_g > 0
    c_occ_ref = c_mo_g[:, mask].real
    c_occ_ref = c_occ_ref.reshape(nao * nimg, -1)

    fock_g = numpy.dot(c_mo_g * e_mo_g, c_mo_g.T.conj())

    from lno.tools.k2gamma import _mat_imag_err
    err_imag = _mat_imag_err(fock_g) / nkpt
    log.info('K2Gamma: F_g.imag= %.1e', err_imag)
    if err_imag > tol_fock_imag:
        log.error('Constructed Gamma-point Fock matrix has large imaginary part %.1e',
                    err_imag)
        raise RuntimeError
    fock_g = fock_g.real
    
    from lno.tools.k2gamma import k2gamma_aoint
    ovlp_k = kmf.get_ovlp()
    ovlp_g = k2gamma_aoint(kmf, ovlp_k)

    hcore_k = kmf.get_hcore()
    hcore_g = k2gamma_aoint(kmf, hcore_k)
    veff_g = fock_g - hcore_g

    # make the parent class more flexible
    from pyscf.pbc.scf.hf import RHF
    class K2Gamma(RHF):
        _is_from_k2gamma = True
        def get_ovlp(self, *args, **kwargs):
            return ovlp_g
        
        def get_hcore(self, *args, **kwargs):
            return hcore_g
        
        def get_veff(self, *args, **kwargs):
            rdm1_g_inp = args[1]
            rdm1_g_ref = self._rdm1_g_ref
            assert numpy.allclose(rdm1_g_inp, rdm1_g_ref)
            return veff_g

        def kernel(self, *args, **kwargs):
            raise RuntimeError('K2Gamma: kernel shall not be called')

    mf_g = K2Gamma(scell)
    e_mo_g, c_mo_g = mf_g.eig(fock_g, ovlp_g)
    mf_g.mo_energy = e_mo_g
    mf_g.mo_coeff = c_mo_g
    mf_g.mo_occ = mf_g.get_occ()
    
    mask = mf_g.mo_occ > 0
    c_occ_sol = c_mo_g[:, mask]
    c_occ_ref = c_occ_ref.reshape(nao * nimg, -1)
    
    # check if the occupied orbitals are the same
    s = c_occ_sol.T @ ovlp_g @ c_occ_ref
    ds = scipy.linalg.det(s)

    if not numpy.allclose(ds, 1):
        s = scipy.linalg.svd(s, full_matrices=False)[1]
        warn = 'occupied orbitals do not span the same space, determinant = % 6.4e\n' % ds
        log.warn(warn)

    mf_g._rdm1_g_ref = mf_g.make_rdm1()
    return mf_g
lno.tools.k2gamma = k2gamma

import lno.base.klno
from lno.base.klno import get_korb
from lno.base.klno import _LNOERIS
def _common_init_(eris, mcc):
    _LNOERIS._common_init_(eris, mcc)
    orbo, orbv = mcc.split_mo()[1:3]
    korbo = get_korb(mcc, orbo)
    korbv = get_korb(mcc, orbv)
    eris.korbo = korbo
    eris.korbv = korbv
    eris.with_df = mcc._kscf.with_df

def get_eris_gen(eris, u, kind):
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)

    mol = df_obj.cell
    kpts = df_obj.kpts
    nkpt = len(kpts)
    nao = mol.nao_nr()

    coeff_occ_kpt = eris.korbo.reshape(nkpt, nao, -1)
    coeff_vir_kpt = eris.korbv.reshape(nkpt, nao, -1)

    coeff = []
    shape = []

    for ic, c in enumerate([coeff_occ_kpt, coeff_vir_kpt, coeff_occ_kpt, coeff_vir_kpt]):
        is_capital = kind[ic].upper() == kind[ic]
        cc = c @ u if is_capital else c
        coeff.append(cc)
        shape.append(cc.shape[-1])

    eris_ovov = df_obj.ao2mo_spc(coeff, kpts=kpts)
    eris_ovov = eris_ovov.reshape(shape)
    return eris_ovov

lno.base.klno._KLNODFINCOREERIS._common_init_ = _common_init_
lno.base.klno._KLNODFINCOREERIS.get_eris_gen = get_eris_gen

import lno.cc.kccsd
def _make_isdf_eris_outcore(cc, mo_coeff=None):
    kmf = cc._kscf
    kpts = kmf.kpts
    nkpts = len(kpts)
    df_obj = kmf.with_df

    from lno.cc.kccsd import _KChemistsERIs
    eris = _KChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir * (nvir + 1) // 2

    Ukw = cc.Ukw
    coeff_mo_spc = eris.mo_coeff.reshape(nkpts, -1, nmo)
    coeff_mo_kpt = numpy.einsum('kw,wmp->kmp', Ukw, coeff_mo_spc)

    eri = df_obj.ao2mo_spc([coeff_mo_kpt] * 4, kpts=kpts)
    eri = eri.reshape([nmo, ] * 4) / 3
    eris_ovov_sol = eri[:nocc, nocc:, :nocc, nocc:]
    eris_ovov_sol = eris_ovov_sol.reshape(nocc * nvir, nocc * nvir)

    from pyscf import lib
    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri.create_dataset(
        'ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc)
        )
    eris.ovov = eris.feri.create_dataset(
        'ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir)
        )
    eris.ovvo = eris.feri.create_dataset(
        'ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc)
        )
    eris.oovv = eris.feri.create_dataset(
        'oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir)
        )
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
    eris.vvvv = eris.feri.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')

    eris.oooo[:] = eri[:nocc, :nocc, :nocc, :nocc]
    eris.ovoo[:] = eri[:nocc, nocc:, :nocc, :nocc]
    eris.ovov[:] = eri[:nocc, nocc:, :nocc, nocc:]
    eris.ovvo[:] = eri[:nocc, nocc:, nocc:, :nocc]
    eris.oovv[:] = eri[:nocc, :nocc, nocc:, nocc:]

    ovvv = eri[:nocc, nocc:, nocc:, nocc:].reshape(-1, nvir, nvir)
    ovvv_pair = lib.pack_tril(ovvv)
    eris.ovvv[:] = ovvv_pair.reshape(nocc, nvir, nvir_pair)

    vvvv = eri[nocc:, nocc:, nocc:, nocc:].reshape(-1, nvir, nvir)
    vvvv_pair = lib.pack_tril(vvvv).reshape(-1, nvir_pair)
    vvvv_pair = vvvv_pair.T.reshape(nvir_pair, nvir, nvir)
    vvvv_pair = lib.pack_tril(vvvv_pair).reshape(nvir_pair, nvir_pair)
    eris.vvvv[:] = vvvv_pair
    return eris
lno.cc.kccsd._make_df_eris_outcore = _make_isdf_eris_outcore