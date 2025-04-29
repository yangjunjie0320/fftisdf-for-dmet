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
    
    basis = ['gth-szv-molopt-sr', 'gth-dzvp-molopt-sr']
    kmesh = ['1-1-2', '1-2-2', '2-2-2', '2-2-3', '2-3-3', '3-3-3', '3-3-4', '3-4-4', '4-4-4']
    ke_cutoff = [50, 200]
    method = ['gdf', 'fftdf', 'fftisdf-10', 'fftisdf-40']

    from itertools import product
    for k, b, m in product(kmesh, basis, method):
        if m == 'gdf':
            ke = 0.0
            dir_name = f"{k}/{b}/{m}/"
            dir_path = base_dir / dir_name
            cmd = f"python main.py --kmesh={k} --basis={b} --ke_cutoff={ke} --method={m}\n"
            name = dir_name
            yield dir_path, cmd, name

        else:
            for ke in ke_cutoff:
                dir_name = f"{k}/{b}/{m}/{ke}/"
                dir_path = base_dir / dir_name
                cmd = f"python main.py --kmesh={k} --basis={b} --ke_cutoff={ke} --method={m}\n"
                name = dir_name
                yield dir_path, cmd, name

def main():
    base_dir = Path(__file__).parent
    for dir_path, cmd, name in loop(base_dir):
        print(f"Setting up benchmark directory: {dir_path}")
        print(f"cmd: {cmd[:-1]}")

        dir_path.mkdir(parents=True, exist_ok=False)
        time = '20:00:00'
        ncpu = 1

        base = Path(__file__).parent
        run_content = None
        with open(base / '../../src/script/run.sh', 'r') as f:
            run_content = f.readlines()
            run_content.insert(1, f"#SBATCH --time={time}\n")
            run_content.insert(1, f"#SBATCH --mem-per-cpu=10gb\n")
            run_content.insert(1, f"#SBATCH --cpus-per-task=32\n")
            run_content.insert(1, f"#SBATCH --ntasks={ncpu}\n")
            run_content.insert(1, f"#SBATCH --job-name={name}\n")

        python_path = [Path(__file__).parent / '../../src/fftisdf-main',
                       Path(__file__).parent / '../../src/libdmet2-main']
        python_path = ":".join([str(p) for p in python_path])
        run_content.append(f"export PYTHONPATH=$PYTHONPATH:{python_path}\n")
        run_content.append(f"cp {base / '../../src/script/main-nio-afm.py'} {dir_path / 'main.py'}\n")
        run_content.append(f"{cmd}\n")
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
