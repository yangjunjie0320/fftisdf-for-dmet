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
    log = new_logger(mp_obj, mp_obj.verbose)

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

def kmp2_corr_energy_with_isdf(mp_obj, nw=20, sos_factor=1.3):
    log = new_logger(mp_obj, mp_obj.verbose)
    
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
    go_kpt = [w ** 0.25 * numpy.exp( e_kpt[k][:nocc, None] * x) for k in range(nkpt)]
    gv_kpt = [w ** 0.25 * numpy.exp(-e_kpt[k][nocc:, None] * x) for k in range(nkpt)]
    go_kpt = numpy.array(go_kpt).reshape(nkpt, nocc, nw)
    gv_kpt = numpy.array(gv_kpt).reshape(nkpt, nvir, nw)

    xo_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, :nocc]) for k in range(nkpt)]
    xo_kpt = numpy.array(xo_kpt).reshape(nkpt, nip, nocc)
    xv_kpt = [numpy.dot(inpv_kpt[k], c_kpt[k, :, nocc:]) for k in range(nkpt)]
    xv_kpt = numpy.array(xv_kpt).reshape(nkpt, nip, nvir)

    from fft.isdf import get_phase_factor, spc_to_kpt, kpt_to_spc
    phase = get_phase_factor(cell, kpts)

    e_corr_os = 0.0
    for iw in range(nw):
        t0 = (process_clock(), perf_counter())
        x_kpt = [lib.dot(x, g[:, None] * x.T.conj()) for x, g in zip(xo_kpt, go_kpt[:, :, iw])]
        y_kpt = [lib.dot(x, g[:, None] * x.T.conj()) for x, g in zip(xv_kpt, gv_kpt[:, :, iw])]
        x_spc = kpt_to_spc(numpy.asarray(x_kpt), phase)
        y_spc = kpt_to_spc(numpy.asarray(y_kpt), phase)
        x_kpt = y_kpt = None

        d_spc = x_spc * y_spc
        d_kpt = spc_to_kpt(d_spc, phase) / numpy.sqrt(nkpt)

        j_kpt = [lib.dot(d.conj(), j) for d, j in zip(d_kpt, coul_kpt)]
        e_corr_os -= numpy.einsum("qKJ,qJK->", j_kpt, j_kpt).real / nkpt

        log.timer("KMP2 iw = %3d" % iw, *t0)
    e_corr = e_corr_os * sos_factor

    log.info("e_corr_os  = % 12.8f" % e_corr_os)
    log.info("e_corr_sos = % 12.8f" % e_corr)
    return e_corr

if __name__ == "__main__":
    kmesh = [1, 1, 3]
    basis = 'gth-dzvp'

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pbe'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.ke_cutoff = 50
    cell.exp_to_discard = 0.1
    cell.build(dump_input=False)

    kpts = cell.make_kpts(kmesh)

    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = fft.ISDF(cell, kpts)
    kmf.with_df.build(cisdf=20.0)
    kmf.exxdiv = None
    kmf.verbose = 0
    ehf = kmf.kernel()
    print("ehf = % 12.8f" % ehf)

    from pyscf.pbc import mp
    kmp = mp.KMP2(kmf)
    kmp.verbose = 0
    kmp.kernel()
    ene_corr_kmp2 = kmp.e_corr
    ene_corr_os = kmp.e_corr_os
    print("ene_corr_kmp2 = % 12.8f" % ene_corr_kmp2)
    print("ene_corr_os   = % 12.8f" % ene_corr_os)

    polw_kpt = krpa_pol_with_isdf(kmp, nw=20)
    e_corr_krpa = krpa_corr_energy_with_isdf(kmp, nw=20, polw_kpt=polw_kpt)
    print("e_corr_krpa = % 12.8f" % e_corr_krpa)

    e_corr_kmp2 = kmp2_corr_energy_with_isdf(kmp, nw=41)
    print("e_corr_kmp2 = % 12.8f" % e_corr_kmp2)
