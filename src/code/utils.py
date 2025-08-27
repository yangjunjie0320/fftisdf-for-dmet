import os, pathlib
import time, h5py
import numpy, scipy
from typing import Optional

import pyscf
from pyscf.lib import H5TmpFile
from pyscf.pbc import gto

TMPDIR = os.environ.get("PYSCF_TMPDIR", None)
assert os.path.exists(TMPDIR)

MAX_MEMORY = os.environ.get("PYSCF_MAX_MEMORY", 2000)
MAX_MEMORY = int(MAX_MEMORY)
assert MAX_MEMORY > 0

def parse_basis(cell: gto.Cell, basis_name: Optional[str] = None):
    if basis_name is None:
        basis_name = "cc-pvdz"
    
    pwd = pathlib.Path(__file__).parent
    basis_path = pwd / "../../data/basis"
    assert basis_path.exists(), f"Path {basis_path} not found"

    basis_file = basis_path / ("%s.dat" % basis_name)
    assert basis_file.exists(), f"Path {basis_file} not found"
    
    uniq_atoms = {a[0] for a in cell.format_atom(cell.atom, unit="Bohr")}

    basis = {}
    for atom_symbol in uniq_atoms:
        from pyscf.gto.basis.parse_nwchem import load
        print(f"Loading basis for {atom_symbol} from {basis_file}")
        basis[atom_symbol] = load(basis_file, atom_symbol)

    return basis

def build_cell(config: dict):
    name: str = config["name"].lower()

    pwd = pathlib.Path(__file__).parent
    poscar_path = pwd / f"../../data/vasp/{name}.vasp"
    poscar_path = poscar_path.resolve()
    poscar_path = poscar_path.absolute()
    print(f"Poscar path: {poscar_path}")
    assert poscar_path.exists(), f"Path {poscar_path} not found"

    from libdmet.utils.iotools import read_poscar
    cell = read_poscar(poscar_path)
    cell.basis = parse_basis(cell, config["basis"])
    cell.pseudo = "gth-hf-rev"
    cell.ke_cutoff = None
    cell.max_memory = MAX_MEMORY
    cell.build(dump_input=False)
    config["cell"] = cell

    kmesh = [int(i) for i in config["kmesh"].split("-")]

    import libdmet
    from libdmet.system.lattice import Lattice
    latt = Lattice(cell, kmesh)
    config["lattice"] = latt

    kmesh = [int(i) for i in config["kmesh"].split("-")]
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    assert numpy.allclose(kpts, latt.kpts)

    config["kmesh"] = kmesh
    config["kpts"] = kpts
    
    from pyscf.pbc.tools.k2gamma import get_phase
    scell, phase = get_phase(cell, kpts)
    config["scell"] = scell

def build_density_fitting(config: dict):
    cell: gto.Cell = config["cell"]
    scell: gto.Cell = config["scell"]
    latt: Lattice = config["lattice"]
    kpts: numpy.ndarray = config["kpts"]

    method = config["density_fitting_method"].lower()
    df_to_read = config["df_to_read"]
    df_to_read = None if df_to_read == "None" else df_to_read

    df_obj = None
    df_path = os.path.join(TMPDIR, "df.h5")
    print("DF path:", df_path)
    
    if "gdf" in method:
        print("Using GDF, method = %s" % method)
        method = method.split("-")

        from pyscf.pbc.df import GDF
        df_obj = GDF(cell, kpts)
        df_obj._cderi_to_save = df_path

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
        df_obj._cderi_to_save = df_path

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
        print(f"ke_cutoff = {cell.ke_cutoff}, mesh = {df_obj.mesh}")

        if df_to_read is not None:
            print(f"FFTDF is not using df_to_read = {df_to_read}")

    elif "fftisdf" in method:
        print("Using FFTISDF, method = %s" % method)
        method = method.split("-")
        assert len(method) == 3, f"Invalid method: {method}"
        
        import fft, fft.isdf_ao2mo
        cell.ke_cutoff = float(method[1])
        cell.build(dump_input=False)

        df_obj = fft.ISDF(cell, kpts)
        print(f"ke_cutoff = {cell.ke_cutoff}, mesh = {df_obj.mesh}")

        df_obj.verbose = 5
        df_obj._isdf_to_save = df_path

        if df_to_read is not None:
            print(f"Reading FFTISDF from {df_to_read}")
            df_obj._isdf = df_to_read
        
        cisdf = float(method[2])
        build_isdf_obj = df_obj.build
        df_obj.build = lambda *args, **kwargs: build_isdf_obj(cisdf=cisdf)
        
        print(f"Using ke_cutoff = {cell.ke_cutoff}, cisdf = {cisdf}")
    
    df_obj.verbose = 5
    config["df"] = df_obj
    config["df_path"] = df_path

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
        alph_label = ["0 Ni 3dz\^2", "0 Ni 3dx2-y2"]
        beta_label = ["1 Ni 3dz\^2", "1 Ni 3dx2-y2"]

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
    mf.exxdiv = "ewald"
    mf.chkfile = "scf.chk"
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

if __name__ == "__main__":
    import sys
    libdmet_path = pathlib.Path(__file__).parent / "../libdmet2-main"
    assert libdmet_path.exists(), f"Path {libdmet_path} not found"
    sys.path.append(str(libdmet_path))

    config = {
        "name": "diamond",
        "kmesh": "1-1-1",
        "basis": "cc-pvdz",
        "density_fitting_method": "gdf-2.0",
        "init_guess_method": "minao",
        "df_to_read": None,
        "xc": None, "is_unrestricted": False,
    }
    build(config)

    config = {
        "name": "silicon",
        "kmesh": "1-1-1",
        "basis": "cc-pvdz",
        "density_fitting_method": "gdf-2.0",
        "init_guess_method": "minao",
        "df_to_read": None,
        "xc": None,
        "is_unrestricted": False,
    }
    build(config)

    config = {
        "name": "co2",
        "kmesh": "1-1-1",
        "basis": "cc-pvdz",
        "density_fitting_method": "gdf-2.0",
        "init_guess_method": "minao",
        "df_to_read": None,
        "xc": None,
        "is_unrestricted": False,
    }

    build(config)

    config = {
        "name": "nh3",
        "kmesh": "1-1-1",
        "basis": "cc-pvdz",
        "density_fitting_method": "gdf-2.0",
        "init_guess_method": "minao",
        "df_to_read": None,
        "xc": None,
        "is_unrestricted": False,
    }
    build(config)