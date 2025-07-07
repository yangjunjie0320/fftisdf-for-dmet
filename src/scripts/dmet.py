import fft, libdmet, numpy, scipy
import pyscf
from pyscf import lib
from pyscf.lib.logger import process_clock, perf_counter

from libdmet.basis.trans_2e_helper import eri_restore

def get_emb_eri_fftisdf_v1(
    mydf, C_ao_lo=None, C_lo_eo=None, unit_eri=False,
    symmetry=4, t_reversal_symm=True, max_memory=None,
    kscaled_center=None, kconserv_tol=1e-12, fname=None
    ):
    print("Get Embedding ERI from FFT-ISDF")

    assert t_reversal_symm

    cell = mydf.cell
    kpts = mydf.kpts

    c_ao_lo_k = C_ao_lo
    c_lo_eo_s = C_lo_eo
    spin, nimg, nlo, neo = c_lo_eo_s.shape
    nkpt = len(kpts)
    nao = cell.nao_nr()
    assert nkpt == nimg

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    assert not unit_eri

    from libdmet.basis.trans_1e import multiply_basis
    from libdmet.basis.trans_2e import get_basis_k
    from libdmet.system.fourier import get_phase_R2k
    phase = get_phase_R2k(cell, kpts)
    c_lo_eo_k = get_basis_k(c_lo_eo_s, phase)
    c_ao_eo_k = multiply_basis(c_ao_lo_k, c_lo_eo_k) / (nkpt ** 0.75)
    assert c_ao_eo_k.shape == (spin, nkpt, nao, neo)

    log = lib.logger.new_logger(mydf)
    t0 = (process_clock(), perf_counter())

    import fft, fft.isdf_ao2mo
    assert isinstance(mydf, fft.ISDF)
    sp = spin * (spin + 1) // 2 # spin pair

    eri_emb = []
    for s1, s2 in [(0, 0), (1, 1), (0, 1)][:sp]:
        c1 = c_ao_eo_k[s1]
        c2 = c_ao_eo_k[s2]
        coeff_kpts = [c1, c1, c2, c2]
        eri_emb_s1_s2 = mydf.ao2mo_spc(coeff_kpts)
        eri_emb_s1_s2 = eri_emb_s1_s2.reshape(neo * neo, neo * neo)
        eri_emb.append(eri_emb_s1_s2.real)

    eri_emb = numpy.asarray(eri_emb)
    eri_emb = eri_emb.reshape(sp, neo * neo, neo * neo)
    eri_emb = eri_restore(eri_emb, symmetry, neo)
    log.timer("Get Embedding ERI from FFT-ISDF V1", *t0)
    return eri_emb

def get_emb_eri_fftisdf_v2(
    mydf, C_ao_lo=None, C_lo_eo=None, unit_eri=False,
    symmetry=4, t_reversal_symm=True, max_memory=None,
    kscaled_center=None, kconserv_tol=1e-12, fname=None
    ):
    log = lib.logger.new_logger(mydf)
    t0 = (process_clock(), perf_counter())

    log.info("Get Embedding ERI from FFT-ISDF")
    log.info("symmetry = %d" % symmetry)
    log.info("t_reversal_symm = %s" % t_reversal_symm)
    log.info("unit_eri = %s" % unit_eri)
    log.info("kscaled_center = %s" % kscaled_center)
    log.info("kconserv_tol = %s" % kconserv_tol)
    log.info("fname = %s" % fname)

    assert symmetry == 4

    assert t_reversal_symm

    cell = mydf.cell
    kpts = mydf.kpts

    c_ao_lo_k = C_ao_lo
    c_lo_eo_s = C_lo_eo
    spin, nspc, nlo, neo = c_lo_eo_s.shape
    nkpt = len(kpts)
    nao = cell.nao_nr()
    neo2 = neo * (neo + 1) // 2
    assert nkpt == nspc

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    assert not unit_eri

    from libdmet.basis.trans_1e import multiply_basis
    from libdmet.basis.trans_2e import get_basis_k
    from libdmet.system.fourier import get_phase_R2k
    phase = get_phase_R2k(cell, kpts)
    c_lo_eo_k = get_basis_k(c_lo_eo_s, phase)
    c_ao_eo_k = multiply_basis(c_ao_lo_k, c_lo_eo_k) / (nkpt ** 0.75)
    assert c_ao_eo_k.shape == (spin, nkpt, nao, neo)

    if t_reversal_symm:
        from libdmet.basis.trans_2e import get_weights_t_reversal
        wkpt = get_weights_t_reversal(cell, kpts)
        log.debug("time reversal symm used, weights of kpts:\n%s", wkpt)
    else:
        wkpt = numpy.ones((nkpt,), dtype=int)
        log.debug("time reversal symm not used.")

    import fft, fft.isdf_ao2mo
    assert isinstance(mydf, fft.ISDF)
    sp = spin * (spin + 1) // 2 # spin pair
    nkpt_ibz = numpy.sum(wkpt != 0)

    coul_kpt = mydf.coul_kpt
    nip = coul_kpt.shape[1]
    inpv_kpt = []
    for s in range(spin):
        xs = [lib.dot(xk, ck) for xk, ck in zip(mydf.inpv_kpt, c_ao_eo_k[s])]
        inpv_kpt.append(numpy.asarray(xs))
    inpv_kpt = numpy.asarray(inpv_kpt).reshape(spin, nkpt, -1)
    
    from fft.isdf_ao2mo import kpt_to_spc, spc_to_kpt
    inpv_spc = [kpt_to_spc(inpv_kpt[s], phase) for s in range(spin)]
    inpv_spc = numpy.asarray(inpv_spc).reshape(spin, nspc * nip, neo)
    
    from pyscf.lib import current_memory
    max_memory = max(2000, mydf.max_memory - current_memory()[0])
    max_memory = max_memory * 1e6 * 0.5

    from pyscf.lib import H5TmpFile
    fswp = H5TmpFile()
    rho_spc = fswp.create_dataset("rho_spc", (spin, nspc * nip, neo2), dtype='float64')
    rho_kpt = fswp.create_dataset("rho_kpt", (spin, nkpt * nip, neo2), dtype='complex128')
    for s in range(spin):
        blksize = max_memory // (8 * neo * neo)
        blksize = max(blksize, 1000)

        log.debug("\nblksize = %d, nspc * nip = %d" % (blksize, nspc * nip))
        log.debug("nspc = %d, nip = %d, neo = %d" % (nspc, nip, neo))
        log.debug("memory required for each block: %d GB" % (8 * neo * neo * blksize / 1e9))
        log.debug("max_memory: %d GB" % (max_memory / 1e9))

        for i0, i1 in lib.prange(0, nspc * nip, blksize):
            x_spc_i0i1 = inpv_spc[s, i0:i1, :].reshape(-1, neo)
            rho_spc_i0i1 = x_spc_i0i1[:, :, None] * x_spc_i0i1[:, None, :]
            rho_spc_i0i1 = rho_spc_i0i1.reshape(i1 - i0, neo, neo)

            from pyscf.lib.numpy_helper import pack_tril
            rho_spc[s, i0:i1] = pack_tril(rho_spc_i0i1)
            x_spc_i0i1 = rho_spc_i0i1 = None
        
        blksize = max_memory // (16 * nkpt * nip)
        blksize = max(blksize, 1000)
        log.debug("\nblksize = %d, nkpt * nip = %d" % (blksize, nkpt * nip))
        log.debug("memory required for each block: %d GB" % (16 * nkpt * nip * blksize / 1e9))
        log.debug("max_memory: %d GB" % (max_memory / 1e9))

        for n0, n1 in lib.prange(0, neo2, blksize):
            rho_spc_n0n1 = rho_spc[s, :, n0:n1]
            rho_spc_n0n1 = rho_spc_n0n1.reshape(nspc, -1)
            rho_kpt_n0n1 = spc_to_kpt(rho_spc_n0n1, phase)
            rho_kpt[s, :, n0:n1] = rho_kpt_n0n1.reshape(-1, n1 - n0)
            rho_spc_n0n1 = rho_kpt_n0n1 = None

    t1 = log.timer("prepare rho_kpts", *t0)

    eri_emb = []
    for s1, s2 in [(0, 0), (1, 1), (0, 1)][:sp]:
        eri_emb_s1_s2 = numpy.zeros((neo2, neo2))
        for q in range(nkpt):
            t0 = (process_clock(), perf_counter())
            if wkpt[q] <= 0:
                continue
            
            i0, i1 = q * nip, (q + 1) * nip
            rho1_q = rho_kpt[s1, i0:i1].reshape(nip, neo2)
            rho2_q = rho_kpt[s2, i0:i1].reshape(nip, neo2)

            vq = lib.dot(rho1_q.T, coul_kpt[q])
            eri_emb_q = lib.dot(vq, rho2_q.conj())
            rho1_q = rho2_q = vq = None

            eri_emb_s1_s2 += eri_emb_q.real * wkpt[q] / (nkpt * nkpt)
            eri_emb_q = None
            log.timer("s1: %d, s2: %d, w[q]: %6.2f, q: %3d / %3d" % (s1, s2, wkpt[q], q + 1, nkpt), *t0)

        eri_emb.append(eri_emb_s1_s2)
        eri_emb_s1_s2 = None

    eri_emb = numpy.asarray(eri_emb)
    eri_emb = eri_emb.reshape(sp, neo2, neo2)
    eri_emb = eri_restore(eri_emb, symmetry, neo)
    log.timer("prepare eri_emb", *t1)
    return eri_emb

get_emb_eri_fftisdf = get_emb_eri_fftisdf_v2

def build_dmet(mf, latt, is_unrestricted):
    from libdmet.lo.make_lo import get_iao
    res = get_iao(mf, minao="scf", full_return=True)
    c_ao_lo_k = res[0]
    idx_val, idx_vir  = res[2:]
    latt.build(idx_val=idx_val, idx_virt=idx_vir)

    from libdmet.solver import cc_solver, fci_solver 
    beta = 1000.0
    kwargs = {
        "restricted": (not is_unrestricted),
        "restart": False, "tol": 1e-6,
        "verbose": 5, "max_cycle": 100,
    }
    solver = cc_solver.CCSD(**kwargs)
    
    from libdmet.dmet import rdmet, udmet
    kwargs = {"solver_argss": [["C_lo_eo"]], "vcor": None}
    emb = rdmet.RDMET(latt, mf, solver, c_ao_lo_k, **kwargs)
    if is_unrestricted:
        emb = udmet.UDMET(latt, mf, solver, c_ao_lo_k, **kwargs)

    emb.dump_flags()  # Print settings information
    emb.beta = beta
    emb.fit_method = 'CG'
    emb.fit_kwargs = {"test_grad": False}
    emb.max_cycle = 1
    return emb
