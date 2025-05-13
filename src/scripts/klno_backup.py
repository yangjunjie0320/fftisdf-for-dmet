import numpy, scipy
import pyscf, fft, lno
import fft.isdf_ao2mo

import lno.base.klno
from lno.base.klno import get_korb
from lno.base.klno import _LNOERIS

class _KLNODFINCOREERIS(lno.base.klno._KLNODFINCOREERIS):
    def _common_init_(eris, mcc):
        _LNOERIS._common_init_(eris, mcc)
        orbo, orbv = mcc.split_mo()[1:3]

        korbo = get_korb(mcc, orbo)
        korbv = get_korb(mcc, orbv)
        eris.ovov = [korbo, korbv, korbo, korbv]

        df_obj = mcc._kscf.with_df
        eris.with_df = df_obj
        assert isinstance(df_obj, fft.ISDF)

    def get_eris_gen(eris, u, kind):
        df_obj = eris.with_df
        assert isinstance(df_obj, fft.ISDF)

        mol = df_obj.cell
        kpts = df_obj.kpts
        nkpt = len(kpts)
        nao = mol.nao_nr()
        
        ovov = []
        for ix, cx in enumerate(eris.ovov):
            is_capital = kind[ix].upper() == kind[ix]
            ovov.append(cx @ u if is_capital else cx)
        
        shape = [cx.shape[-1] for cx in ovov]
        eris_ovov = df_obj.ao2mo_spc(ovov, kpts=kpts)
        eris_ovov = eris_ovov.reshape(shape)
        return eris_ovov / nkpt
    
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


def _make_isdf_eris_outcore(cc, mo_coeff=None):
    phase = cc.Ukw
    nimg, nkpt = phase.shape
    assert nimg == nkpt
    
    kmf = cc._kscf
    kpts = kmf.kpts
    cell = kmf.cell
    df_obj = kmf.with_df
    assert isinstance(df_obj, fft.ISDF)

    nao_per_img = cell.nao_nr()
    nao = nao_per_img * nimg

    from lno.cc.kccsd import _KChemistsERIs
    eris = _KChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir * (nvir + 1) // 2

    coeff_mo_spc = eris.mo_coeff.reshape(nimg, nao_per_img, nmo)
    coeff_mo_kpt = numpy.einsum('kw,wmp->kmp', phase, coeff_mo_spc)

    eri = df_obj.ao2mo_spc([coeff_mo_kpt] * 4, kpts=kpts)
    eri = eri.reshape([nmo, ] * 4) / nkpt

    from pyscf import lib
    eris.feri = lib.H5TmpFile()
    shape = (nocc, nocc, nocc, nocc)
    eris.oooo = eris.feri.create_dataset('oooo', shape, 'f8')
    eris.oooo = eri[:nocc, :nocc, :nocc, :nocc]
    
    shape = (nocc, nvir, nocc, nocc)
    chunks = (nocc, 1, nocc, nocc)
    eris.ovoo = eris.feri.create_dataset('ovoo', shape, 'f8', chunks=chunks)
    eris.ovoo = eri[:nocc, nocc:, :nocc, :nocc]

    shape = (nocc, nvir, nocc, nvir)
    chunks = (nocc, 1, nocc, nvir)
    eris.ovov = eris.feri.create_dataset('ovov', shape, 'f8', chunks=chunks)
    eris.ovov = eri[:nocc, nocc:, :nocc, nocc:]

    shape = (nocc, nvir, nvir, nocc)
    chunks = (nocc, 1, nvir, nocc)
    eris.ovvo = eris.feri.create_dataset('ovvo', shape, 'f8', chunks=chunks)
    eris.ovvo = eri[:nocc, nocc:, nocc:, :nocc]

    shape = (nocc, nocc, nvir, nvir)
    chunks = (nocc, nocc, 1, nvir)
    eris.oovv = eris.feri.create_dataset('oovv', shape, 'f8', chunks=chunks)
    eris.oovv = eri[:nocc, :nocc, nocc:, nocc:]

    shape = (nocc, nvir, nvir_pair)
    eris.ovvv = eris.feri.create_dataset('ovvv', shape, 'f8')
    eris_ovvv = eri[:nocc, nocc:, nocc:, nocc:].reshape(-1, nvir, nvir)
    eris_ovvv = lib.pack_tril(eris_ovvv)
    eris.ovvv = eris_ovvv.reshape(nocc, nvir, nvir_pair)
    eris_ovvv = None

    shape = (nvir_pair, nvir_pair)
    eris.vvvv = eris.feri.create_dataset('vvvv', shape, 'f8')
    eris_vvvv = eri[nocc:, nocc:, nocc:, nocc:].reshape(-1, nvir, nvir)
    eris_vvvv = lib.pack_tril(eris_vvvv).T
    eris_vvvv = eris_vvvv.reshape(nvir_pair, nvir, nvir)
    eris_vvvv = lib.pack_tril(eris_vvvv)
    eris.vvvv = eris_vvvv.reshape(nvir_pair, nvir_pair)
    eris_vvvv = None
    eri = None

    return eris

def K2GCCSD(kmf, frozen=None, mo_coeff=None, mo_occ=None, mf=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf import scf

    if mf is None:
        mf = k2gamma(kmf)
        mf.with_df = kmf.with_df
        mf.exxdiv = None
        assert isinstance(kmf.with_df, fft.ISDF)

    is_from_k2gamma = getattr(mf, '_is_from_k2gamma', False)
    assert is_from_k2gamma

    if mo_occ is None:
        mo_occ = mf.mo_occ

    assert mo_coeff is not None
    assert mo_occ is not None

    from lno.cc.kccsd import _K2GCCSD
    class _K2GCCSD(lno.cc.kccsd._K2GCCSD):
        def ao2mo(self, mo_coeff=None):
            return _make_isdf_eris_outcore(self, mo_coeff)
        
    return _K2GCCSD(kmf, mf, frozen, mo_coeff, mo_occ)

import lno.cc.kccsd
class KLNOCCSD(lno.cc.kccsd.KLNOCCSD):
    def _k2g_common_init_(self, kmf, mf=None):
        from lno.tools import get_k2g_phase
        self._kscf = kmf

        if mf is None:
            mf = k2gamma(kmf).rs_density_fit()
            mf.with_df.build = lambda *args, **kwargs: None
        is_from_k2gamma = getattr(mf, '_is_from_k2gamma', False)
        assert is_from_k2gamma

        self.Ukw, self.kmesh = get_k2g_phase(kmf.cell, kmf.kpts)[1:]
        if getattr(kmf, 'with_df', None):
            self.with_df = kmf.with_df
        else:
            raise RuntimeError
        self._keys.update(['with_df','_kscf','Ukw','kmesh'])
        return mf
        
    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris, frozen=None, log=None):
        # remember that everthing here is in the supercell world
        mo_occ = mf.mo_occ
        nmo = mo_occ.size

        from lno.cc.ccsd import get_maskact
        frozen, maskact = get_maskact(frozen, nmo)

        mcc = K2GCCSD(self._kscf, mf=mf, mo_coeff=mo_coeff, frozen=frozen)
        mcc.set(verbose=self.verbose_imp)

        if eris is not None:
            mcc._s1e = eris.s1e
            mcc._h1e = eris.h1e
            mcc._vhf = eris.vhf

        assert not self.ccsd_t

        assert hasattr(mcc, '_s1e')
        assert hasattr(mcc, '_h1e')
        assert hasattr(mcc, '_vhf')

        from lno.cc.ccsd import impurity_solve
        res = impurity_solve(
            mcc, mo_coeff, lo_coeff, mo_occ, maskact, eris, log=log,
            ccsd_t=self.ccsd_t, verbose_imp=self.verbose_imp
            )
        return res  

    def ao2mo(self):
        eris = _KLNODFINCOREERIS()
        eris._common_init_(self)
        return eris
