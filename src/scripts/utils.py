import os
import time, h5py
import numpy, scipy

import pyscf, libdmet
from pyscf.lib import H5TmpFile
from pyscf.pbc import gto
from libdmet.system.lattice import Lattice
from libdmet.lo.make_lo import get_iao

def dump(config : dict, path : str):
    raise NotImplementedError

def build_cell(config: dict):
    name: str = config["name"]

    atom = None
    a = None
    if "diamond" in name.lower():
        pass
    elif "nio" in name.lower():
        pass

    assert atom is not None, f"Unknown cell: {name}"
    cell = gto.Cell()
    cell.atom, cell.a = atom, a
    cell.basis = config["basis"]
    cell.pseudo = config["pseudo"]
    cell.ke_cutoff = config["ke_cutoff"]
    cell.exp_to_discard = 0.1
    cell.build(dump_input=False)
    config["cell"] = cell

    kmesh = [int(i) for i in config.get("kmesh", "1-1-1").split("-")]
    latt = Lattice(cell, kmesh)
    config["lattice"] = latt

def build_density_fitting(config: dict):
    cell: gto.Cell = config.get("cell")
    latt: Lattice = config.get("lattice")
    kpts: numpy.ndarray = latt.kpts
    kmesh: list[int] = latt.kmesh

    method = config.get("density-fitting-method", "GDF")
    df_to_read = config.get("df-to-read", None)

    df_obj = None
    if "gdf" in method.lower():
        from pyscf.pbc.df import GDF
        df_obj = GDF(cell, kpts)

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
        df_obj._fswap = H5TmpFile()
        df_obj.tol = 1e-8
        df_obj.wrap_around = True

        m0 = cell.cutoff_to_mesh(50.0)
        g0 = cell.gen_uniform_grids(m0)
        c0 = float(method[1])
        inpx = df_obj.select(g0=g0, c0=c0, kpts=kpts, tol=1e-30)

        df_build = df_obj.build
        df_obj.build = lambda: df_build(inpx=inpx)

    assert df_obj is not None, f"Unknown density fitting method: {method}"
    config["df"] = df_obj

def get_init_guess(config: dict):
    name: str = config["name"]
    alph_label = []
    beta_label = []

    if "nio-afm" in name.lower():
        alph_label = ["Ni"]
        beta_label = ["O"]

    mf: pyscf.pbc.scf.kscf.KSCF = config.get("mf")
    is_unrestricted: bool = config.get("is_unrestricted", False)
    is_spin_polarized = len(alph_label) > 0 or len(beta_label) > 0

    spin = 2 if is_unrestricted else 1
    nao = mf.cell.nao_nr()

    init_guess_method = config.get("init_guess_method", "minao")
    dm0 = mf.get_init_guess(key=init_guess_method)
    dm0 = dm0.reshape(spin, nao, nao)

    if is_spin_polarized:
        assert is_unrestricted

        print("Preparing initial guess for spin polarized calculation")
        print(f"Alph label: {alph_label}, Index: {alph_ix}")
        print(f"Beta label: {beta_label}, Index: {beta_ix}")

        alph_ix = mf.cell.search_ao_label(alph_label)
        beta_ix = mf.cell.search_ao_label(beta_label)

        dm0[0, beta_ix, beta_ix] *= 0.0
        dm0[1, alph_ix, alph_ix] *= 0.0
    
    config["dm0"] = dm0

def build_mean_field(config: dict):
    xc = config["xc"]
    is_unrestricted = config["is_unrestricted"]

    cell: gto.Cell = config["cell"]
    latt: Lattice = config["lattice"]
    kpts: numpy.ndarray = latt.kpts
    kmesh: list[int] = latt.kmesh

    mf = pyscf.pbc.scf.KRHF(cell, kpts)
    if is_unrestricted:
        mf = mf.to_uhf()

    if xc is not None:
        raise NotImplementedError
    
    from libdmet.mean_field import pbc_helper as pbc_hp
    config["mf"] = pbc_hp.smearing_(mf, sigma=0.01)
    get_init_guess(config)

def build_dmet(config: dict):
    latt = config["lattice"]
    mf = config["mf"]
    is_unrestricted = config["is_unrestricted"]
    
    res = get_iao(mf, minao="scf", full_return=True)
    c_ao_lo_k = res[0]
    idx_core, idx_vale = res[1:]
    latt.build(idx_core=idx_core, idx_val=idx_vale)

    from libdmet.dmet import cc_solver
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
    emb.mu_glob = 0.2
    if is_unrestricted:
        emb = udmet.UDMET(latt, mf, solver, c_ao_lo_k, **kwargs)
        emb.mu_glob = [0.2, 0.2]

    emb.dump_flags()  # Print settings information
    emb.beta = beta
    emb.fit_method = 'CG'
    emb.fit_kwargs = {"test_grad": False}
    emb.max_cycle = 1
    config["emb"] = emb

def build(config):
    # for cases it is a argparse.Namespace
    if not isinstance(config, dict):
        config = config.__dict__
    
    config = config.copy(deep=True)
    build_cell(config)
    build_density_fitting(config)
    build_mean_field(config)
    build_dmet(config) # doing nothing for now
    return config

import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--kmesh", type=str, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--pseudo", type=str, required=True)
    parser.add_argument("--ke-cutoff", type=float, required=True)
    parser.add_argument("--is-unrestricted", type=bool, default=False)
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--density-fitting-method", type=str, default="GDF")
    parser.add_argument("--df-to-read", type=str, default=None)
    config = parser.parse_args().__dict__

    print("Running %s with Config:" % (__file__))
    for k, v in config.items():
        print(f"{k}: {v}")

    return config
