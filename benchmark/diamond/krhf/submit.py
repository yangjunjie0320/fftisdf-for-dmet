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
    kmesh = ['1-1-2', '1-2-2', '2-2-2', '2-2-3', '2-3-3', '3-3-3', '3-3-4', '3-4-4', '4-4-4']
    method = ['rsdf', 'gdf', 'fftdf', 'fftisdf']

    from itertools import product
    for k, b, m in product(kmesh, basis, method):
        config = {'kmesh': k, 'basis': b}

        if m in ['rsdf', 'gdf']:
            for beta in [1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]:
                config['density-fitting-method'] = "%s-%s" % (m, beta)
                dir_name = f"{k}/{b}/{m}/{beta}"
                dir_path = base_dir / dir_name
                yield dir_path, config

        elif m == 'fftdf':
            for ke_cutoff in [50, 100, 150, 200]:
                config['density-fitting-method'] = "%s-%s" % (m, ke_cutoff)
                dir_name = f"{k}/{b}/{m}/{ke_cutoff}"
                dir_path = base_dir / dir_name
                yield dir_path, config

        else:
            assert m == 'fftisdf'
            for ke_cutoff in [50, 100, 150, 200]:
                for c in [5, 10, 15, 20, 25, 30]:
                    config['density-fitting-method'] = "%s-%s-%s" % (m, ke_cutoff, c)
                    dir_name = f"{k}/{b}/{m}/{ke_cutoff}/{c}"
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
