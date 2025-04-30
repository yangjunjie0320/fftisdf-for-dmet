import os
import time, h5py
import numpy, scipy

import pyscf, libdmet
from pyscf.lib import H5TmpFile
from pyscf.pbc import gto
from libdmet.system.lattice import Lattice
from libdmet.lo.make_lo import get_iao

TMPDIR = os.environ.get("PYSCF_TMPDIR", None)
assert os.path.exists(TMPDIR)

MAX_MEMORY = os.environ.get("PYSCF_MAX_MEMORY", 2000)
MAX_MEMORY = int(MAX_MEMORY)
assert MAX_MEMORY > 0


def dump(config : dict, path : str):
    raise NotImplementedError

def build_cell(config: dict):
    name: str = config["name"]

    atom = None
    a = None
    if "diamond" in name.lower():
        atom = '''
        C 0.0000 0.0000 0.0000
        C 0.8917 0.8917 0.8917
        '''
        
        a = '''
        0.0000 1.7834 1.7834
        1.7834 0.0000 1.7834
        1.7834 1.7834 0.0000
        '''

    elif "nio" in name.lower():
        atom = '''
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

    assert atom is not None, f"Unknown cell: {name}"
    cell = gto.Cell()
    cell.atom, cell.a = atom, a
    cell.basis = config["basis"]
    cell.pseudo = config["pseudo"]
    cell.ke_cutoff = config["ke_cutoff"]
    cell.exp_to_discard = 0.1
    cell.max_memory = MAX_MEMORY
    cell.build(dump_input=False)
    config["cell"] = cell

    kmesh = [int(i) for i in config["kmesh"].split("-")]
    latt = Lattice(cell, kmesh)
    config["lattice"] = latt

def build_density_fitting(config: dict):
    cell: gto.Cell = config["cell"]
    latt: Lattice = config["lattice"]
    kpts: numpy.ndarray = latt.kpts
    kmesh: list[int] = latt.kmesh

    method = config["density_fitting_method"]
    df_to_read = config["df_to_read"]
    df_to_read = None if df_to_read == "None" else df_to_read

    df_obj = None
    if "gdf" in method.lower():
        from pyscf.pbc.df import GDF
        df_obj = GDF(cell, kpts)
        df_obj.exclude_dd_block = False
        df_obj.exxdiv = None
        df_obj._prefer_ccdf = True

        if df_to_read is not None:
            assert os.path.exists(df_to_read)
            df_obj._cderi = df_to_read

    if "fftdf" in method.lower():
        from pyscf.pbc.df import FFTDF
        df_obj = FFTDF(cell, kpts)

    elif "fftisdf" in method.lower():
        method = method.lower().split("-")
        assert len(method) == 2, f"Invalid method: {method}"
        
        import fft
        df_obj = fft.ISDF(cell, kpts)
        df_obj.tol = 1e-8
        df_obj._fswap = H5TmpFile()
        df_obj.wrap_around = True
        df_obj.verbose = 5

        m0 = cell.cutoff_to_mesh(50.0)
        g0 = cell.gen_uniform_grids(m0)
        c0 = float(method[1])
        inpx = df_obj.select_inpx(g0=g0, c0=c0, kpts=kpts, tol=1e-30)

        df_build = df_obj.build
        df_obj.build = lambda: df_build(inpx=inpx)

    assert df_obj is not None, f"Unknown density fitting method: {method}"
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
    latt: Lattice = config["lattice"]
    kpts: numpy.ndarray = latt.kpts

    mf = pyscf.pbc.scf.KRHF(cell, kpts)
    mf.verbose = 5
    mf.conv_tol = 1e-6
    mf.exxdiv = None
    if is_unrestricted:
        mf = mf.to_uhf()

    if xc is not None:
        raise NotImplementedError
    
    from libdmet.mean_field import pbc_helper as pbc_hp
    config["mf"] = pbc_hp.smearing_(mf, sigma=0.01)
    get_init_guess(config)

def build(config):
    # for cases it is a argparse.Namespace
    if not isinstance(config, dict):
        config = config.__dict__
    
    build_cell(config)
    build_density_fitting(config)
    build_mean_field(config)
    return config
