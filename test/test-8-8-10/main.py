import os, sys, numpy, time
import pyscf, libdmet, fft, utils

def main(config: dict):
    from utils import build
    build(config)
    
    cell = config["cell"]
    kmesh  = ['1-1-1', '1-1-2', '1-2-2', '2-2-2']
    kmesh += ['2-2-3', '2-3-3', '3-3-3']
    kmesh += ['3-3-4', '3-4-4', '4-4-4']
    kmesh += ['4-4-5', '4-5-5', '5-5-5']
    kmesh += ['5-5-6', '5-6-6', ]
    kmesh += ['6-6-6']
    kmesh += ['6-6-7', '6-7-7', '7-7-7']
    kmesh += ['7-7-8', '7-8-8', '8-8-8']
    kmesh += ['8-8-10', '8-10-10', '10-10-10']

    from pyscf.lib.chkfile import dump
    for km in kmesh:
        m = [int(i) for i in km.split("-")]
        kpts = cell.make_kpts(m, wrap_around=True)
        nk = len(kpts)
        
        t0 = time.time()
        from fft.isdf import get_kconserv
        kconserv = get_kconserv(cell, kpts)
        t1 = time.time()
        dt = (t1 - t0) / 3600
        print("nk = %4d, time_get_kconserv = % 8.2e h" % (nk, dt), flush=True)
        
        path = f"{config['name']}-kconserv-wrap-around-1.chk"
        dump(path, km + "/kpts", kpts)
        dump(path, km + "/kconserv3", kconserv)
        dump(path, km + "/kconserv2", kconserv[:, :, 0].T)

    for km in kmesh:
        m = [int(i) for i in km.split("-")]
        kpts = cell.make_kpts(m, wrap_around=False)
        nk = len(kpts)
        
        t0 = time.time()
        from fft.isdf import get_kconserv
        kconserv = get_kconserv(cell, kpts)
        t1 = time.time()
        dt = (t1 - t0) / 3600
        print("nk = %4d, time_get_kconserv = % 8.2e h" % (nk, dt), flush=True)
        
        path = f"{config['name']}-kconserv-wrap-around-0.chk"
        dump(path, km + "/kpts", kpts)
        dump(path, km + "/kconserv3", kconserv)
        dump(path, km + "/kconserv2", kconserv[:, :, 0].T)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--kmesh", type=str, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--pseudo", type=str, required=True)
    parser.add_argument("--lno-thresh", type=float, default=3e-5)
    parser.add_argument("--density-fitting-method", type=str, required=True)
    parser.add_argument("--is-unrestricted", action="store_true")
    parser.add_argument("--init-guess-method", type=str, default="minao")
    parser.add_argument("--df-to-read", type=str, default=None)
    parser.add_argument("--kconserv-to-read", type=str, default=None)
    
    print("\nRunning %s with:" % (__file__))
    config = parser.parse_args().__dict__
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")

    main(config)
