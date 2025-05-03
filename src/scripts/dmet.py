import fft, libdmet, numpy, scipy
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
    return eri_emb

def get_emb_eri_fftisdf_v2(
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
    spin, nspc, nlo, neo = c_lo_eo_s.shape
    nkpt = len(kpts)
    nao = cell.nao_nr()
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

    import fft, fft.isdf_ao2mo
    assert isinstance(mydf, fft.ISDF)
    sp = spin * (spin + 1) // 2 # spin pair

    nip = mydf._inpv_kpt.shape[1]
    inpv_kpts = [mydf._inpv_kpt @ c_ao_eo_k[s] for s in range(spin)]
    inpv_kpts = numpy.asarray(inpv_kpts).reshape(spin, nkpt, -1)
    coul_kpt = mydf._coul_kpt

    from fft.isdf_ao2mo import kpt_to_spc, spc_to_kpt
    inpv_spcs = [kpt_to_spc(inpv_kpts[s], phase) for s in range(spin)]
    inpv_spcs = numpy.asarray(inpv_spcs).reshape(spin, nspc, nip, neo)

    rho_kpts = []
    for s in range(spin):
        x_s = inpv_spcs[s]
        rho_spc = x_s.reshape(nspc * nip, -1, 1) * x_s.reshape(nspc * nip, 1, -1)
        rho_spc = rho_spc.reshape(nspc, nip, -1)
        rho_kpt = spc_to_kpt(rho_spc, phase)
        rho_kpt = rho_kpt.reshape(nkpt, nip, -1)
        rho_kpts.append(rho_kpt)

    rho_kpts = numpy.asarray(rho_kpts).reshape(spin, nkpt, nip, -1)
    print(rho_kpts.shape, "the size of rho_kpts is ", rho_kpts.nbytes / 1e9, "GB")

    eri_emb = []
    for s1, s2 in [(0, 0), (1, 1), (0, 1)][:sp]:
        rho1_kpt = rho_kpts[s1]
        rho2_kpt = rho_kpts[s2]
        n1 = rho1_kpt.shape[2]
        n2 = rho2_kpt.shape[2]
        eri_emb_s1_s2 = numpy.zeros((n1, n2))
        for q in range(nkpt):
            rho1_q = rho1_kpt[q].reshape(nip, n1)
            rho2_q = rho2_kpt[q].reshape(nip, n2)
            eri_emb_q = rho1_q.T @ coul_kpt[q] @ rho2_q.conj()
            eri_emb_s1_s2 += eri_emb_q.real / (nkpt * nkpt)
        eri_emb.append(eri_emb_s1_s2)
    eri_emb = numpy.asarray(eri_emb)
    eri_emb = eri_emb.reshape(sp, neo * neo, neo * neo)
    eri_emb = eri_restore(eri_emb, symmetry, neo)
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
