import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path

def loop(method="krhf"):
    basis = 'gth-dzvp'
    kmesh  = ['2-2-2', '2-2-3', '2-3-3', '3-3-3', '3-3-4', '3-4-4', '4-4-4', '4-4-6']
    kmesh += ['4-6-6', '6-6-6', '6-6-8', '6-8-8', '8-8-8', '8-8-10', '8-10-10', '10-10-10']
    thresh = ["1e-6", "3e-7", "1e-7", "3e-8", "1e-8"]
    df_method = ['gdf-2.0', 'fftdf-100']

    from itertools import product
    config = None
    if method == "klno":
        config = [{'basis': basis, 'kmesh': k, 'lno-thresh': t, 'density-fitting-method': d} for k, t, d in product(kmesh, thresh, df_method)]
    else:
        config = [{'basis': basis, 'kmesh': k, 'density-fitting-method': d} for k, d in product(kmesh, df_method)]

    def make_dir_name(c):
        n = ""
        if c.get('basis', None):
            n += c['basis'] + "/"
        if c.get('kmesh', None):
            n += c['kmesh'] + "/"
        if c.get('density-fitting-method', None):
            n += c['density-fitting-method'] + "/"
        if c.get('lno-thresh', None):
            n += method + "-" + c['lno-thresh'] + "/"
        else:
            n += method + "/"
        return n

    for c in config:
        d = make_dir_name(c)
        yield d, c


def main(cell='diamond', method='krhf', ntasks=1, time='20:00:00'):
    base_dir = Path(__file__).parent

    for dir_name, config in loop(method=method):
        print(f"Setting up benchmark directory: {dir_name}")

        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=False)

        config['name'] = cell
        config['pseudo'] = 'gth-hf'
        config['is-unrestricted'] = False
        config['init-guess-method'] = 'minao'

        base = Path(__file__).parent
        run_content = None

        src_path = base / '../../src/'
        src_path = src_path.resolve()
        src_path = src_path.absolute()
        
        job_name = str(dir_path).split(cell)[1].split('/')
        job_name = cell + '-'.join(job_name)

        with open(src_path / 'scripts/run.sh', 'r') as f:
            run_content = f.readlines()
            run_content.insert(1, f"#SBATCH --time={time}\n")
            run_content.insert(1, f"#SBATCH --mem-per-cpu=10gb\n")
            run_content.insert(1, f"#SBATCH --cpus-per-task=32\n")
            run_content.insert(1, f"#SBATCH --ntasks={ntasks}\n")
            run_content.insert(1, f"#SBATCH --job-name={job_name}\n")
            run_content.insert(1, f"#SBATCH --reservation=changroup_standingres\n")

        # convert to absolute path
        python_path  = [src_path / 'fftisdf-main', src_path / 'libdmet2-main']
        python_path += [src_path / 'pyscf-forge-lnocc', src_path / 'scripts']

        run_content.append(f"export PYTHONPATH={python_path[0]}\n")
        for p in python_path[1:]:
            run_content.append(f"export PYTHONPATH=$PYTHONPATH:{p}\n")

        # construct the main script path, and make sure it exists
        main_path = src_path / ('scripts/main-%s.py' % method)
        assert main_path.exists(), f"Main script not found: {main_path}"
        run_content.append(f"cp {main_path} ./main.py\n")

        is_unrestricted = config.pop('is-unrestricted')
        cmd = "python main.py %s" % " ".join([f"--{k}={v}" for k, v in config.items()])
        if is_unrestricted:
            cmd += " --is-unrestricted"
        run_content.append(cmd + "\n")
        run_content.append(f"echo \"End time = $(date)\"\n")

        with open(dir_path / 'run.sh', 'w') as f:
            f.write("".join(run_content))

        # run the run.sh
        os.chdir(dir_path)
        subprocess.run(["sbatch", "run.sh"])
        os.chdir(Path(__file__).parent)
        print(f"Submitted job for {dir_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond")
    parser.add_argument("--method", type=str, default="krhf")
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--time", type=str, default="20:00:00")
    args = parser.parse_args()
    kwargs = args.__dict__
    for k, v in kwargs.items():
        print(f"{k}: {v}")
    print()
    main(**kwargs)
