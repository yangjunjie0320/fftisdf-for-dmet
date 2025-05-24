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
    ke_cutoff = [100, 200]
    method = ['rsdf', 'fftisdf-10', 'fftisdf-20']

    from itertools import product
    for k, b, m in product(kmesh, basis, method):
        config = {'kmesh': k, 'basis': b, 'density-fitting-method': m}
        if m == 'rsdf':
            config['ke-cutoff'] = 0.0
            dir_name = f"{k}/{b}/{m}/"
            dir_path = base_dir / dir_name
            yield dir_path, config

        else:
            for ke in ke_cutoff:
                config['ke-cutoff'] = ke
                dir_name = f"{k}/{b}/{m}/{ke}/"
                dir_path = base_dir / dir_name
                yield dir_path, config

def main():
    base_dir = Path(__file__).parent
    name = base_dir.name
    name = name.split('-')[0]
    for dir_path, config in loop(base_dir):
        print(f"Setting up benchmark directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=False)

        config['name'] = name
        config['pseudo'] = 'gth-pbe'
        config['is-unrestricted'] = False
        config['init-guess-method'] = 'minao'
        config['df-to-read'] = None

        time = '04:00:00'
        ntasks = 1

        base = Path(__file__).parent
        run_content = None

        src_path = base / '../../src/'
        src_path = src_path.resolve()
        src_path = src_path.absolute()

        with open(src_path / 'scripts/run.sh', 'r') as f:
            run_content = f.readlines()
            run_content.insert(1, f"#SBATCH --time={time}\n")
            run_content.insert(1, f"#SBATCH --mem-per-cpu=10gb\n")
            run_content.insert(1, f"#SBATCH --cpus-per-task=32\n")
            run_content.insert(1, f"#SBATCH --ntasks={ntasks}\n")
            run_content.insert(1, f"#SBATCH --job-name={name}\n")
            run_content.insert(1, f"#SBATCH --reservation=changroup_standingres\n")

        # convert to absolute path
        python_path  = [src_path / 'fftisdf-main', src_path / 'libdmet2-main']
        python_path += [src_path / 'pyscf-forge-lnocc', src_path / 'scripts']

        run_content.append(f"export PYTHONPATH={python_path[0]}\n")
        for p in python_path[1:]:
            run_content.append(f"export PYTHONPATH=$PYTHONPATH:{p}\n")
        run_content.append(f"cp {(src_path / 'scripts/main-krhf-lno.py')} ./main.py\n")

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
    main()
