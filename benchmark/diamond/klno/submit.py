import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path

def loop(base_dir):
    """Create benchmark directories for different parameter combinations."""
    # Create parent directory first
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    basis = ['gth-dzvp']
    kmesh  = ['1-1-2', '1-2-2', '2-2-2', '2-2-3', '2-3-3', '3-3-3', '3-3-4', '3-4-4', '4-4-4', '4-4-6']
    kmesh += ['4-6-6', '6-6-6', '6-6-8', '6-8-8', '8-8-8', '8-8-10', '8-10-10', '10-10-10']
    # method = ['rsdf', 'fftisdf']
    thresh = ["1e-4", "1e-6", "1e-8", "1e-9", "1e-10"]

    from itertools import product
    # for k, b in product(kmesh, basis):
    #     config = {'kmesh': k, 'basis': b}

    for k, b, t in product(kmesh, basis, thresh):
        config = {'kmesh': k, 'basis': b, 'lno-thresh': t}

        m = "rsdf"
        beta = 2.0
        config['density-fitting-method'] = "%s-%s" % (m, beta)
        dir_name = f"{k}/{b}/{m}/{beta}/{t}"
        dir_path = base_dir / dir_name
        yield dir_path, config

        m = 'fftisdf'
        ke_cutoff = 200
        c = 20
        config['density-fitting-method'] = "%s-%s-%s" % (m, ke_cutoff, c)
        dir_name = f"{k}/{b}/{m}/{ke_cutoff}/{c}/{t}"
        dir_path = base_dir / dir_name
        yield dir_path, config

def main(cell='diamond', method='krhf', ntasks=1, time='20:00:00'):
    base_dir = Path(__file__).parent

    for dir_path, config in loop(base_dir):
        print(f"Setting up benchmark directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=False)

        config['name'] = cell
        config['pseudo'] = 'gth-pbe'
        config['is-unrestricted'] = False
        config['init-guess-method'] = 'minao'

        base = Path(__file__).parent
        run_content = None

        src_path = base / '../../../src/'
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
