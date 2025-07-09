import os, sys, numpy, time
import mpi4py
comm = mpi4py.MPI.COMM_WORLD

import pyscf, libdmet, fft
import libdmet.basis.trans_2e

def main(config: dict):
    from utils import build
    build(config)

    table = {}
    t0 = time.time()
    cell = config["cell"]
    kpts = config["lattice"].kpts
    method = config["density_fitting_method"]
    method = method.lower().split("-")
    assert method[0] == "fftisdf"

    from fft.isdf_mpi import ISDF
    df_obj = fft.isdf_mpi.ISDF(cell, kpts, comm=comm)
    df_obj.tol = 1e-8
    df_obj.wrap_around = True
    df_obj.verbose = 5

    m0 = cell.cutoff_to_mesh(50.0)
    g0 = cell.gen_uniform_grids(m0)
    c0 = float(method[1])
    inpx = df_obj.select_inpx(g0=g0, c0=c0, kpts=kpts, tol=1e-30)
    df_obj.build(inpx=inpx)
    table["time_build_df"] = time.time() - t0
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--kmesh", type=str, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--pseudo", type=str, required=True)
    parser.add_argument("--ke-cutoff", type=float, required=True)
    parser.add_argument("--density-fitting-method", type=str, required=True)
    parser.add_argument("--is-unrestricted", action="store_true")
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--df-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)
