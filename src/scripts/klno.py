import numpy, scipy
import pyscf, fft
from pyscf import lib
import pyscf.lno, pyscf.pbc.lno

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
    
    dm_oo = numpy.zeros((nocc, nocc), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('Ia+Jb->IJab', e_ov, e_ov)
    dm_oo += 4 * lib.einsum('IJac,IJbc->ab', t2_oovv, t2_oovv.conj())
    dm_oo -= 2 * lib.einsum('IJac,IJcb->ab', t2_oovv, t2_oovv.conj())
    return dm_oo

def make_lo_rdm1_occ_2p(eris, moeocc, moevir, u):
    log = lib.logger.new_logger(eris)
    assert u.dtype == numpy.float64
    
    df_obj = eris.with_df
    assert isinstance(df_obj, fft.ISDF)
    assert isinstance(eris, _KLNODFINCOREERIS)
    
    nocc, nvir = eris.nocc, eris.nvir
    nvir_lo = u.shape[1]

    from pyscf.lno.make_lno_rdm1 import subspace_eigh
    e_occ = moeocc
    e_vir = moevir

    f = numpy.diag(e_vir)
    e_vir_lo, u_vir_lo = subspace_eigh(f, u)
    assert u_vir_lo.shape == (nvir, nvir_lo)
    
    ovov = eris.get_eris_gen(u_vir_lo, 'oVoV')
    assert ovov.shape == (nocc, nvir_lo, nocc, nvir_lo)

    e_ov = e_occ[:, None] - e_vir_lo
    assert e_ov.shape == (nocc, nvir_lo)

    dm_vv = numpy.zeros((nvir, nvir), dtype=numpy.float64)
    t2_oovv = ovov.transpose(0, 2, 1, 3)
    t2_oovv = t2_oovv / lib.direct_sum('iA+jB->ijAB', e_ov, e_ov)
    dm_vv -= 4 * lib.einsum('ikAB,jkAB->ij', t2_oovv.conj(), t2_oovv)
    dm_vv += 2 * lib.einsum('ikAB,jkAB->ij', t2_oovv.conj(), t2_oovv)
    return dm_vv

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
    
    def make_lo_rdm1_occ(self, eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type):
        assert occ_lno_type in ['1h', '1p', '2p']
        if occ_lno_type == '2p':
            return make_lo_rdm1_occ_2p(eris, moeocc, moevir, uvir_loc)
        else:
            raise NotImplementedError
    
    def make_lo_rdm1_vir(self, eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type):
        assert vir_lno_type in ['1h', '1p', '2h']
        if vir_lno_type == '2h':
            return make_lo_rdm1_vir_2h(eris, moeocc, moevir, uocc_loc)
        else:
            raise NotImplementedError

import pyscf.pbc.lno.klno
def KLNOCCSD(kmf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None, mf=None):
    df_obj = kmf.with_df
    if not isinstance(df_obj, fft.ISDF):
        from pyscf.pbc.df.rsdf import RSDF
        assert isinstance(df_obj, RSDF)
        return pyscf.pbc.lno.klnoccsd.KLNOCCSD(kmf, lo_coeff, frag_lolist, lno_type, lno_thresh, frozen, mf)
    
    return WithFFTISDF(kmf, lo_coeff, frag_lolist, lno_type, lno_thresh, frozen, mf)
