import sys, os, itertools
from itertools import product, repeat

import numpy, scipy

import pyscf
from pyscf import lib
from pyscf.lib.logger import new_logger
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc import gto, scf, df
from pyscf.gw.gw_ac import _get_scaled_legendre_roots
from pyscf.gw.rpa import _get_clenshaw_curtis_roots

import fft

def krpa_pol_with_isdf(mp_obj, nw=20):
    fswap = getattr(mp_obj, '_fswap', None)
    if fswap is None:
        fswap = lib.H5TmpFile()
        mp_obj._fswap = fswap
    
    assert fswap is not None

    kmf_obj = mp_obj._scf
    cell = kmf_obj.cell
    nao = cell.nao_nr()
    kpts = kmf_obj.kpts
    nkpt = len(kpts)

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    kconserv3 = get_kconserv(cell, kpts)
    kconserv2 = kconserv3[:, :, 0].T

    e_kpt = numpy.array(kmf_obj.mo_energy)
    c_kpt = numpy.array(kmf_obj.mo_coeff)
    n_kpt = numpy.array(kmf_obj.mo_occ)
    nmo = mp_obj.nmo
    nocc = mp_obj.nocc
    nvir = nmo - nocc
    nov = nocc * nvir

    assert e_kpt.shape == (nkpt, nmo)
    assert c_kpt.shape == (nkpt, nao, nmo)
    assert n_kpt.shape == (nkpt, nmo)

    df_obj = kmf_obj.with_df
    assert isinstance(df_obj, fft.ISDF)
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)

    xo_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, :nocc]) for k in range(nkpt)]
    xo_kpt = numpy.array(xo_kpt).reshape(nkpt, nip, nocc)
    xv_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, nocc:]) for k in range(nkpt)]
    xv_kpt = numpy.array(xv_kpt).reshape(nkpt, nip, nvir)

    polw_kpt = fswap.create_dataset('polw_kpt', (nw, nkpt, nip, nip), dtype=numpy.complex128)

    for ifreq, (freq, weig) in enumerate(zip(*_get_scaled_legendre_roots(nw))):
        for q in range(nkpt):
            pol_f_q = numpy.zeros((nip, nip), dtype=numpy.complex128)
            for (ki, ka) in product(range(nkpt), repeat=2):
                if not kconserv2[ki, ka] == q:
                    continue

                eov = e_kpt[ki, :nocc, None] - e_kpt[ka, None, nocc:]
                dov = eov / (freq ** 2 + eov ** 2)
                dov = dov.reshape(nov)
                
                xi = xo_kpt[ki].conj().reshape(nip, nocc, 1)
                xa = xv_kpt[ka].reshape(nip, 1, nvir)
                rov = xi * xa
                rov = rov.reshape(nip, nov)
                
                lov = rov * dov
                pol_f_q += lib.dot(lov, rov.conj().T) * 4 / nkpt
            polw_kpt[ifreq, q] = pol_f_q
            pol_f_q = None
    return polw_kpt

def krpa_corr_energy_with_isdf(mp_obj, nw=20, polw_kpt=None):

    log = new_logger(mp_obj, 5)

    fswap = getattr(mp_obj, '_fswap', None)
    assert fswap is not None
    
    kmf_obj = mp_obj._scf
    cell = kmf_obj.cell
    nao = cell.nao_nr()
    kpts = kmf_obj.kpts
    nkpt = len(kpts)

    kconserv3 = kmf_obj.with_df.kconserv3
    kconserv2 = kmf_obj.with_df.kconserv2

    e_kpt = numpy.array(kmf_obj.mo_energy)
    c_kpt = numpy.array(kmf_obj.mo_coeff)
    n_kpt = numpy.array(kmf_obj.mo_occ)
    nmo = mp_obj.nmo
    nocc = mp_obj.nocc
    nvir = nmo - nocc
    nov = nocc * nvir
    assert e_kpt.shape == (nkpt, nmo)
    assert c_kpt.shape == (nkpt, nao, nmo)
    assert n_kpt.shape == (nkpt, nmo)

    df_obj = kmf_obj.with_df
    assert isinstance(df_obj, fft.ISDF)
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)

    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    e_corr = 0.0
    
    for q, kq in enumerate(kpts):
        t0 = (process_clock(), perf_counter())
        coul_q = coul_kpt[q]

        for ifreq, (freq, weig) in enumerate(zip(*_get_scaled_legendre_roots(nw))):
            polw_q = polw_kpt[ifreq, q]
            pq = lib.dot(coul_q, polw_q.T)

            dq = numpy.linalg.det(numpy.eye(nip) - pq)
            e_corr_wq = numpy.log(dq) + numpy.trace(pq)
            e_corr += e_corr_wq.real * weig / 2 / numpy.pi / nkpt
            polw_q = None

        coul_q = None
        log.timer("RPA q = %d" % q, *t0)

    fswap.close()
    return e_corr

def kmp2_corr_energy_with_isdf(mp_obj, nw=20, polw_kpt=None):
    log = new_logger(mp_obj, 5)

    fswap = getattr(mp_obj, '_fswap', None)
    assert fswap is not None
    
    kmf_obj = mp_obj._scf
    cell = kmf_obj.cell
    nao = cell.nao_nr()
    kpts = kmf_obj.kpts
    nkpt = len(kpts)

    kconserv3 = kmf_obj.with_df.kconserv3
    kconserv2 = kmf_obj.with_df.kconserv2

    e_kpt = numpy.array(kmf_obj.mo_energy)
    c_kpt = numpy.array(kmf_obj.mo_coeff)
    n_kpt = numpy.array(kmf_obj.mo_occ)
    nmo = mp_obj.nmo
    nocc = mp_obj.nocc
    nvir = nmo - nocc
    nov = nocc * nvir
    assert e_kpt.shape == (nkpt, nmo)
    assert c_kpt.shape == (nkpt, nao, nmo)
    assert n_kpt.shape == (nkpt, nmo)

    df_obj = kmf_obj.with_df
    assert isinstance(df_obj, fft.ISDF)
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)

    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    x, w = _get_clenshaw_curtis_roots(nw)
    gw_kpt = [w ** 0.25 * numpy.exp(e_kpt[k][:, None] * x) for k in range(nkpt)]
    gw_kpt = numpy.array(gw_kpt).reshape(nkpt, nmo, nw)
    go_kpt = gw_kpt[:, :nocc, :]
    gv_kpt = gw_kpt[:, nocc:, :]

    xo_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, :nocc]) for k in range(nkpt)]
    xo_kpt = numpy.array(xo_kpt).reshape(nkpt, nip, nocc)
    xv_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, nocc:]) for k in range(nkpt)]
    xv_kpt = numpy.array(xv_kpt).reshape(nkpt, nip, nvir)

    e_corr = 0.0
    for iw in range(nw):
        tow_kpt = numpy.einsum("kIi,ki,kKi->kIK", xo_kpt, go_kpt[:, :, iw], xo_kpt.conj(), optimize=True)
        tvw_kpt = numpy.einsum("kIa,ka,kKa->kIK", xv_kpt, gv_kpt[:, :, iw], xv_kpt.conj(), optimize=True)

        from fft.isdf import get_phase_factor, spc_to_kpt, kpt_to_spc
        phase = get_phase_factor(kscaled, kpts)
        tow_kpt = spc_to_kpt(tow_kpt, phase)
        tvw_kpt = spc_to_kpt(tvw_kpt, phase)

        tow_spc = kpt_to_spc(tow_kpt, phase)
        tvw_spc = kpt_to_spc(tvw_kpt, phase)

        tw_spc = tow_spc * tvw_spc
        tw_kpt = spc_to_kpt(tw_spc, phase)
        tw_kpt = tw_kpt.conj() * numpy.sqrt(nkpt)

        j_kpt = [lib.dot(coul_kpt[k].conj().T, tw_kpt[k]) for k in range(nkpt)]
        e_corr -= numpy.sum([j_kpt[k].T * j_kpt[k] for k in range(nkpt)]).real / (nkpt ** 3)

    fswap.close()
    return e_corr

if __name__ == "__main__":
    kmesh = [1, 1, 3]
    basis = 'gth-dzvp'

    cell = gto.Cell()
    cell.atom = '''
    C 1.337625 1.337625 1.337625
    C 2.229375 2.229375 2.229375
    '''
    cell.a = '''
    0.000000     1.783500     1.783500
    1.783500     0.000000     1.783500
    1.783500     1.783500     0.000000
    '''
    cell.unit = 'angstrom'
    cell.max_memory = 2000
    cell.ke_cutoff = 50.0
    cell.verbose = 0
    cell.pseudo = 'gth-pbe'
    cell.basis = basis
    cell.build()

    kpts = cell.make_kpts(kmesh)

    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = fft.ISDF(cell, kpts)
    kmf.with_df.build(cisdf=10.0)
    kmf.verbose = 5
    kmf.kernel()

    from pyscf.pbc import mp
    kmp = mp.KMP2(kmf)

    polw_kpt = krpa_pol_with_isdf(kmp, nw=20)
    e_corr_krpa = krpa_corr_energy_with_isdf(kmp, nw=20, polw_kpt=polw_kpt)
    print("e_corr_krpa = % 12.8f" % e_corr_krpa)

    e_corr_kmp2 = kmp2_corr_energy_with_isdf(kmp, nw=20, polw_kpt=polw_kpt)
    print("e_corr_kmp2 = % 12.8f" % e_corr_kmp2)
