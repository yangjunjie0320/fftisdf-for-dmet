import os
import time, h5py
import numpy, scipy

import pyscf
from pyscf.lib import H5TmpFile
from pyscf.pbc import gto

TMPDIR = os.environ.get("PYSCF_TMPDIR", None)
assert os.path.exists(TMPDIR)

MAX_MEMORY = os.environ.get("PYSCF_MAX_MEMORY", 2000)
MAX_MEMORY = int(MAX_MEMORY)
assert MAX_MEMORY > 0

def dump(config : dict, path : str):
    raise NotImplementedError

def build_cell(config: dict):
    name: str = config["name"].lower()

    cell = gto.Cell()
    if "diamond" in name:
        cell.atom = '''
        C 0.0000 0.0000 0.0000
        C 0.8917 0.8917 0.8917
        '''
        
        cell.a = '''
        0.0000 1.7834 1.7834
        1.7834 0.0000 1.7834
        1.7834 1.7834 0.0000
        '''

    elif "nio" in name:
        cell.atom = '''
        Ni 0.000 0.000 0.000
        Ni 4.170 4.170 4.170
        O  2.085 2.085 2.085
        O  6.255 6.255 6.255
        '''

        a = '''
        4.170 2.085 2.085
        2.085 4.170 2.085
        2.085 2.085 4.170
        '''
    
    else:
        raise RuntimeError(f"Unknown cell: {name}")

    cell.basis = config["basis"]
    cell.pseudo = config["pseudo"]
    cell.ke_cutoff = None
    cell.exp_to_discard = 0.1
    cell.max_memory = MAX_MEMORY
    cell.unit = 'A'
    cell.build(dump_input=False)
    config["cell"] = cell

    # kmesh = [int(i) for i in config["kmesh"].split("-")]
    # latt = Lattice(cell, kmesh)
    # config["lattice"] = latt

    kmesh = [int(i) for i in config["kmesh"].split("-")]
    config["kmesh"] = kmesh
    config["kpts"] = cell.make_kpts(kmesh)

def build_density_fitting(config: dict):
    cell: gto.Cell = config["cell"]
    # latt: Lattice = config["lattice"]
    # kpts: numpy.ndarray = latt.kpts
    kpts: numpy.ndarray = config["kpts"]

    method = config["density_fitting_method"].lower()
    df_to_read = config["df_to_read"]
    df_to_read = None if df_to_read == "None" else df_to_read

    df_obj = None
    if "gdf" in method:
        print("Using GDF, method = %s" % method)
        method = method.split("-")

        from pyscf.pbc.df import GDF
        df_obj = GDF(cell, kpts)
        df_obj.exxdiv = None

        if df_to_read is not None:
            assert os.path.exists(df_to_read)
            df_obj._cderi = df_to_read
        
        if len(method) == 2:
            beta = float(method[1])
            print(f"Using beta = {beta}")
            from pyscf.df import aug_etb
            df_obj.auxbasis = aug_etb(cell, beta=beta)
        else:
            print("Using default settings for GDF")
    
    elif "rsdf" in method:
        print("Using RSDF, method = %s" % method)
        method = method.split("-")

        from pyscf.pbc.df import RSDF
        df_obj = RSDF(cell, kpts)
        df_obj.exxdiv = None

        if df_to_read is not None:
            assert os.path.exists(df_to_read)
            df_obj._cderi = df_to_read
        
        if len(method) == 2:
            beta = float(method[1])
            print(f"Using beta = {beta}")
            from pyscf.df import aug_etb
            df_obj.auxbasis = aug_etb(cell, beta=beta)
        else:
            print("Using default settings for GDF")

    elif "fftdf" in method:
        print("Using FFTDF, method = %s" % method)
        method = method.split("-")
        assert len(method) == 2, f"Invalid method: {method}"

        from pyscf.pbc.df import FFTDF
        cell.ke_cutoff = float(method[1])
        cell.build(dump_input=False)

        df_obj = FFTDF(cell, kpts)
        df_obj.exxdiv = None
        print(f"ke_cutoff = {cell.ke_cutoff}, mesh = {df_obj.mesh}")

        if df_to_read is not None:
            print(f"FFTDF is not using df_to_read = {df_to_read}")

    elif "fftisdf" in method:
        print("Using FFTISDF, method = %s" % method)
        method = method.split("-")
        assert len(method) == 3, f"Invalid method: {method}"
        
        import fft
        cell.ke_cutoff = float(method[1])
        cell.build(dump_input=False)

        df_obj = fft.ISDF(cell, kpts)
        df_obj.exxdiv = None
        print(f"ke_cutoff = {cell.ke_cutoff}, mesh = {df_obj.mesh}")

        df_obj.tol = 1e-8
        df_obj.wrap_around = False
        df_obj.verbose = 5
        df_obj.c0 = float(method[2])

        if df_to_read is not None:
            print(f"Reading FFTISDF from {df_to_read}")
            df_obj._isdf = df_to_read
        
        print(f"Using ke_cutoff = {cell.ke_cutoff}, c0 = {df_obj.c0}")

    config["df"] = df_obj

def get_init_guess(config: dict):
    name: str = config["name"]
    mf: pyscf.pbc.scf.kscf.KSCF = config["mf"]
    cell: gto.Cell = config["cell"]
    kpts = mf.kpts
    is_unrestricted: bool = config["is_unrestricted"]

    alph_label = []
    beta_label = []
    alph_ix = []
    beta_ix = []

    if "nio-afm" in name.lower():
        alph_label = ["0 Ni 3dz2", "0 Ni 3dx2-y2"]
        beta_label = ["1 Ni 3dz2", "1 Ni 3dx2-y2"]

        alph_ix = cell.search_ao_label(alph_label)
        beta_ix = cell.search_ao_label(beta_label)

    spin = 2 if is_unrestricted else 1
    is_spin_polarized = len(alph_label) > 0 or len(beta_label) > 0
    nao = mf.cell.nao_nr()
    nkpt = len(kpts)

    init_guess_method = config["init_guess_method"]
    dm0 = mf.get_init_guess(key=init_guess_method)
    dm0 = dm0.reshape(spin, nkpt, nao, nao)

    if is_spin_polarized:
        assert is_unrestricted

        print("Preparing initial guess for spin polarized calculation")
        print(f"Alph label: {alph_label}, Index: {alph_ix}")
        print(f"Beta label: {beta_label}, Index: {beta_ix}")
        dm0[0, :, beta_ix, beta_ix] *= 0.0
        dm0[1, :, alph_ix, alph_ix] *= 0.0
    
    dm0 = dm0[0] if spin == 1 else dm0
    mf.mulliken_meta(cell, dm0)
    config["dm0"] = dm0

def build_mean_field(config: dict):
    xc = config["xc"]
    is_unrestricted = config["is_unrestricted"]

    cell: gto.Cell = config["cell"]
    kpts: numpy.ndarray = config["kpts"]

    mf = pyscf.pbc.scf.KRHF(cell, kpts)
    mf.verbose = 5
    mf.conv_tol = 1e-6
    mf.exxdiv = None
    if is_unrestricted:
        mf = mf.to_uhf()

    if xc is not None:
        raise NotImplementedError
    
    config["mf"] = mf
    get_init_guess(config)

def build(config):
    # for cases it is a argparse.Namespace
    if not isinstance(config, dict):
        config = config.__dict__
    
    build_cell(config)
    build_density_fitting(config)
    build_mean_field(config)
    return config
