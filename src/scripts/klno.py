import numpy, scipy, os
import pyscf, fft
from pyscf import lib
from pyscf.lib import logger

import pyscf.lno, pyscf.pbc.lno
from pyscf.pbc.lno.tools import K2SDF

import fft.isdf_ao2mo

class _KLNODFINCOREERIS(pyscf.pbc.lno.klno._KLNODFINCOREERIS_REAL):
    def __init__(self, with_df, orbocc, orbvir, max_memory=4000, verbose=None, stdout=None):
        super().__init__(with_df, orbocc, orbvir, max_memory, verbose, stdout)
        assert isinstance(with_df, fft.ISDF)

        self._ovov = None

    def build(self):
        if self._ovov is not None:
            return
        
        from pyscf.pbc.lno.tools import s2k_mo_coeff
        orbocc_k = s2k_mo_coeff(self.cell, self.kpts, self.orbocc)
        orbvir_k = s2k_mo_coeff(self.cell, self.kpts, self.orbvir)
        self._ovov = [orbocc_k, orbvir_k, orbocc_k, orbvir_k]

        df_obj = self.with_df
        assert isinstance(df_obj, fft.ISDF)

    def get_eris_gen(self, u, kind):
        df_obj = self.with_df
        assert isinstance(df_obj, fft.ISDF)

        mol = df_obj.cell
        kpts = df_obj.kpts
        nkpt = len(kpts)
        nao = mol.nao_nr()
        
        ovov = []
        for ix, cx in enumerate(self._ovov):
            is_capital = kind[ix].upper() == kind[ix]
            ovov.append(cx @ u if is_capital else cx)
        
        shape = [cx.shape[-1] for cx in ovov]
        eris_ovov = df_obj.ao2mo_spc(ovov, kpts=kpts)
        eris_ovov = eris_ovov.reshape(shape)
        return eris_ovov / nkpt
    
from pyscf.lno.lnoccsd import MODIFIED_CCSD
class MODIFIED_K2SCCSD(MODIFIED_CCSD):
    _keys = {'with_df', 'k2sdf'}
    def __init__(self, mf, with_df, frozen, mo_coeff, mo_occ):
        MODIFIED_CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.with_df = with_df
        self.k2sdf = K2SDF(with_df)
        assert isinstance(with_df, fft.ISDF)
    
    def ao2mo(self, mo_coeff=None):
        from pyscf.lno.lnoccsd import _ChemistsERIs
        log = logger.new_logger(self.with_df)

        eris = _ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        
        mo_coeff = eris.mo_coeff
        nocc = eris.nocc
        nmo = mo_coeff.shape[1]
        nvir = nmo - nocc
        nvir_pair = nvir * (nvir + 1) // 2

        df_obj = self.with_df
        assert isinstance(df_obj, fft.ISDF)

        cell = df_obj.cell
        nao_per_img = cell.nao_nr()
        
        kpts = df_obj.kpts
        nkpt = nimg = len(kpts)
        phase = self.k2sdf.phase
        coeff_mo_spc = eris.mo_coeff.reshape(nimg, nao_per_img, nmo)
        coeff_mo_kpt = numpy.einsum('kw,wmp->kmp', phase.conj(), coeff_mo_spc)

        eri = df_obj.ao2mo_spc([coeff_mo_kpt] * 4, kpts=kpts)
        eri = eri.reshape([nmo, ] * 4) / nkpt

        log.debug('eri.shape = %s', eri.shape)
        log.debug('nocc = %s, nmo = %s, nvir = %s', nocc, nmo, nvir)

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

        log.debug('eris is saved to %s', eris.feri.filename)
        log.debug('file size = %s', os.path.getsize(eris.feri.filename))
        return eris
        

def make_lo_rdm1_vir_2h(eris, moeocc, moevir, u):    
    log = lib.logger.new_logger(eris)
    assert u.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)
    
    nocc, nvir = eris.nocc, eris.nvir
    nocc_lo = u.shape[1]
    
    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    e_occ = moeocc
    e_vir = moevir

    f = numpy.diag(e_occ)
    e_occ_lo, u_occ_lo = subspace_eigh(f, u)
    assert u_occ_lo.shape == (nocc, nocc_lo)
    
    # TODO: (a) with sliced occupied orbitals (or virtual orbital, which is better?)
    #       (b) implement LT formula
    ovov = eris.get_eris_gen(u_occ_lo, 'OvOv')
    assert ovov.shape == (nocc_lo, nvir, nocc_lo, nvir)
    
    e_ov = e_occ_lo[:, None] - e_vir
    assert e_ov.shape == (nocc_lo, nvir)
    
    dm_vv = numpy.zeros((nvir, nvir), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('Ia+Jb->IJab', e_ov, e_ov)
    dm_vv += 4 * lib.einsum('IJac,IJbc->ab', t2_oovv, t2_oovv.conj())
    dm_vv -= 2 * lib.einsum('IJac,IJcb->ab', t2_oovv, t2_oovv.conj())
    return dm_vv

def make_lo_rdm1_vir_1h(eris, e_occ, e_vir, u_occ):
    log = lib.logger.new_logger(eris)
    assert u_occ.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)

    nocc, nvir = eris.nocc, eris.nvir
    nocc_lo = u_occ.shape[1]
    assert u_occ.shape == (nocc, nocc_lo)

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    f_occ = numpy.diag(e_occ)
    e_occ_lo, u_occ_lo = subspace_eigh(f_occ, u_occ)
    assert u_occ_lo.shape == (nocc, nocc_lo)

    ovov = eris.get_eris_gen(u_occ_lo, 'Ovov')
    assert ovov.shape == (nocc_lo, nvir, nocc, nvir)

    e_ov_1 = e_occ_lo[:, None] - e_vir
    e_ov_2 = e_occ[:, None] - e_vir
    assert e_ov_1.shape == (nocc_lo, nvir)
    assert e_ov_2.shape == (nocc, nvir)
    
    dm_vv = numpy.zeros((nvir, nvir), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('Ia+jb->Ijab', e_ov_1, e_ov_2)
    dm_vv += 2 * lib.einsum('Ijac,Ijbc->ab', t2_oovv, t2_oovv.conj())
    dm_vv += 2 * lib.einsum('Ijca,Ijcb->ab', t2_oovv, t2_oovv.conj())
    dm_vv -= lib.einsum('Ijac,Ijcb->ab', t2_oovv, t2_oovv.conj())
    dm_vv -= lib.einsum('Ijca,Ijbc->ab', t2_oovv, t2_oovv.conj())
    return dm_vv

def make_lo_rdm1_vir_1p(eris, e_occ, e_vir, u_vir):
    log = lib.logger.new_logger(eris)
    assert u_vir.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)

    nocc, nvir = eris.nocc, eris.nvir
    nvir_lo = u_vir.shape[1]
    assert u_vir.shape == (nvir, nvir_lo)

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    f_vir = numpy.diag(e_vir)
    e_vir_lo, u_vir_lo = subspace_eigh(f_vir, u_vir)
    assert u_vir_lo.shape == (nvir, nvir_lo)

    ovov = eris.get_eris_gen(u_vir_lo, 'oVov')
    assert ovov.shape == (nocc, nvir_lo, nocc, nvir)

    e_ov_1 = e_occ[:, None] - e_vir_lo
    e_ov_2 = e_occ[:, None] - e_vir
    assert e_ov_1.shape == (nocc, nvir_lo)
    assert e_ov_2.shape == (nocc, nvir)

    dm_vv = numpy.zeros((nvir, nvir), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('iA+jb->ijAb', e_ov_1, e_ov_2)
    dm_vv += 4 * lib.einsum('ijAa,ijAb->ab', t2_oovv.conj(), t2_oovv)
    dm_vv -= 2 * lib.einsum('ijAa,jiAb->ab', t2_oovv.conj(), t2_oovv)
    return dm_vv

def make_lo_rdm1_occ_2p(eris, e_occ, e_vir, u_vir):
    log = lib.logger.new_logger(eris)
    assert u_vir.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)
    
    nocc, nvir = eris.nocc, eris.nvir
    nvir_lo = u_vir.shape[1]
    assert u_vir.shape == (nvir, nvir_lo)

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    f_vir = numpy.diag(e_vir)
    e_vir_lo, u_vir_lo = subspace_eigh(f_vir, u_vir)
    assert u_vir_lo.shape == (nvir, nvir_lo)

    ovov = eris.get_eris_gen(u_vir_lo, 'oVoV')
    assert ovov.shape == (nocc, nvir_lo, nocc, nvir_lo)

    e_ov_1 = e_ov_2 = e_occ[:, None] - e_vir_lo
    assert e_ov_1.shape == (nocc, nvir_lo)
    assert e_ov_2.shape == (nocc, nvir_lo)

    dm_oo = numpy.zeros((nocc, nocc), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('iA+jB->ijAB', e_ov_1, e_ov_2)
    dm_oo -= 4 * lib.einsum('ikAB,jkAB->ij', t2_oovv.conj(), t2_oovv)
    dm_oo += 2 * lib.einsum('ikAB,jkAB->ij', t2_oovv.conj(), t2_oovv)
    return dm_oo

def make_lo_rdm1_occ_1h(eris, e_occ, e_vir, u_occ):
    log = lib.logger.new_logger(eris)
    assert u_occ.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)

    nocc, nvir = eris.nocc, eris.nvir
    nocc_lo = u_occ.shape[1]
    assert u_occ.shape == (nocc, nocc_lo)

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    f_occ = numpy.diag(e_occ)
    e_occ_lo, u_occ_lo = subspace_eigh(f_occ, u_occ)
    assert u_occ_lo.shape == (nocc, nocc_lo)
    
    ovov = eris.get_eris_gen(u_occ_lo, 'ovOv')
    assert ovov.shape == (nocc, nvir, nocc_lo, nvir)

    e_ov_1 = e_occ[:, None] - e_vir
    e_ov_2 = e_occ_lo[:, None] - e_vir
    assert e_ov_1.shape == (nocc, nvir)
    assert e_ov_2.shape == (nocc_lo, nvir)
    
    dm_oo = numpy.zeros((nocc, nocc), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('ia+Jb->iJab', e_ov_1, e_ov_2)
    dm_oo -= 4 * lib.einsum('iKab,jKab->ij', t2_oovv.conj(), t2_oovv)
    dm_oo += 2 * lib.einsum('iKab,jKba->ij', t2_oovv.conj(), t2_oovv)
    return dm_oo

def make_lo_rdm1_occ_1p(eris, e_occ, e_vir, u_vir):
    log = lib.logger.new_logger(eris)
    assert u_vir.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)

    nocc, nvir = eris.nocc, eris.nvir
    nvir_lo = u_vir.shape[1]
    assert u_vir.shape == (nvir, nvir_lo)

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    f_vir = numpy.diag(e_vir)
    e_vir_lo, u_vir_lo = subspace_eigh(f_vir, u_vir)
    assert u_vir_lo.shape == (nvir, nvir_lo)

    ovov = eris.get_eris_gen(u_vir_lo, 'oVov')
    assert ovov.shape == (nocc, nvir_lo, nocc, nvir)

    e_ov_1 = e_occ[:, None] - e_vir_lo
    e_ov_2 = e_occ[:, None] - e_vir
    assert e_ov_1.shape == (nocc, nvir_lo)
    assert e_ov_2.shape == (nocc, nvir)

    dm_oo = numpy.zeros((nocc, nocc), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('iA+jb->ijAb', e_ov_1, e_ov_2)
    dm_oo -= 2 * lib.einsum('ikAb,jkAb->ij', t2_oovv.conj(), t2_oovv)
    dm_oo -= 2 * lib.einsum('kiAb,kjAb->ij', t2_oovv.conj(), t2_oovv)
    dm_oo += lib.einsum('ikAb,kjAb->ij', t2_oovv.conj(), t2_oovv)
    dm_oo += lib.einsum('kiAb,jkAb->ij', t2_oovv.conj(), t2_oovv)
    
    return dm_oo

class WithFFTISDF(pyscf.pbc.lno.klnoccsd.KLNOCCSD):
    def ao2mo(self):
        df_obj = self.with_df
        orb_occ, orb_vir = self.split_mo_coeff()[1:3]
        nocc, nvir = orb_occ.shape[1], orb_vir.shape[1]
        eris = _KLNODFINCOREERIS(
            df_obj, orb_occ, orb_vir,
            max_memory=self.max_memory,
            verbose=self.verbose,
            stdout=self.stdout
        )
        eris.build()
        return eris
    
    def make_lo_rdm1_occ(self, eris, eo, ev, uo, uv, lno_type):
        assert lno_type in ['1h', '1p', '2p']

        dm_oo = None
        if lno_type == '1h':
            dm_oo = make_lo_rdm1_occ_1h(eris, eo, ev, uo)
        elif lno_type == '1p':
            dm_oo = make_lo_rdm1_occ_1p(eris, eo, ev, uv)
        elif lno_type == '2p':
            dm_oo = make_lo_rdm1_occ_2p(eris, eo, ev, uv)

        from pyscf.pbc.lno.make_lno_rdm1 import _check_dm_imag
        assert dm_oo is not None, "Unknown LNO type: %s" % lno_type
        dm_oo = _check_dm_imag(eris, dm_oo)
        return dm_oo
    
    def make_lo_rdm1_vir(self, eris, eo, ev, uo, uv, lno_type):
        assert lno_type in ['1h', '1p', '2h']

        dm_vv = None
        if lno_type == '1h':
            dm_vv = make_lo_rdm1_vir_1h(eris, eo, ev, uo)
        elif lno_type == '1p':
            dm_vv = make_lo_rdm1_vir_1p(eris, eo, ev, uv)
        elif lno_type == '2h':
            dm_vv = make_lo_rdm1_vir_2h(eris, eo, ev, uo)

        from pyscf.pbc.lno.make_lno_rdm1 import _check_dm_imag
        assert dm_vv is not None, "Unknown LNO type: %s" % lno_type
        dm_vv = _check_dm_imag(eris, dm_vv)
        return dm_vv
    
    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        from pyscf.lno.lnoccsd import impurity_solve
        from pyscf.lno.lnoccsd import get_maskact
        from pyscf.pbc.lib.kpts_helper import gamma_point

        if log is None: log = lib.logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, mo_occ.size)

        with_df = self.with_df
        assert isinstance(with_df, fft.ISDF)
        assert gamma_point(with_df.kpts[0]) and numpy.isrealobj(mo_coeff)

        mcc = MODIFIED_K2SCCSD(mf, with_df, frozen, mo_coeff, mo_occ)
        mcc.verbose = self.verbose_imp
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf

        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        res = impurity_solve(
            mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris, log=log,
            ccsd_t=self.ccsd_t, verbose_imp=self.verbose_imp,
            max_las_size_ccsd=self._max_las_size_ccsd,
            max_las_size_ccsd_t=self._max_las_size_ccsd_t
        )
        return res

import pyscf.pbc.lno.klno
def KLNOCCSD(kmf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None, mf=None):
    df_obj = kmf.with_df
    if not isinstance(df_obj, fft.ISDF):
        from pyscf.pbc.df.rsdf import RSDF
        assert isinstance(df_obj, RSDF)
        return pyscf.pbc.lno.klnoccsd.KLNOCCSD(kmf, lo_coeff, frag_lolist, lno_type, lno_thresh, frozen, mf)
    
    return WithFFTISDF(kmf, lo_coeff, frag_lolist, lno_type, lno_thresh, frozen, mf)
