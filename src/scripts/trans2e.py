def get_emb_eri_fftisdf(
    mydf, C_ao_lo=None, C_lo_eo=None, unit_eri=False,
    symmetry=4, t_reversal_symm=True, max_memory=None,
    kscaled_center=None, kconserv_tol=1e-12, fname=None
    ):
    """
    Compute embedding space ERI on the fly, by FFTDF.
    """
    print("Get Embedding ERI from FFT-ISDF")

    assert t_reversal_symm

    cell = mydf.cell
    kpts = mydf.kpts

    c_ao_lo_k = C_ao_lo
    nkpt, nao, nlo = c_ao_lo_k.shape

    c_lo_eo_s = C_lo_eo
    spin, nimg, nlo, neo = c_lo_eo_s.shape

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

    assert spin == 1
    c_ao_eo_k = c_ao_eo_k[0]

    import fft, fft.isdf_ao2mo
    assert isinstance(mydf, fft.ISDF)
    eri_emb = mydf.ao2mo_spc(c_ao_eo_k)
    eri_emb = eri_emb.reshape(spin, neo * neo, neo * neo)

    from libdmet.basis.trans_2e_helper import eri_restore
    eri_emb = eri_restore(eri_emb, symmetry, neo)
    return eri_emb.real


