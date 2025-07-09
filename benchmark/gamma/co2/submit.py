import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path

def loop():
    basis = 'cc-pvdz'
    # df_method  = ['gdf-3.0', 'gdf-2.8', 'gdf-2.6', 'gdf-2.4', 'gdf-2.2', 'gdf-2.0', 'gdf-1.8', 'gdf-1.6', 'gdf-1.4', 'gdf-1.2']
    # df_method += ['fftdf-20', 'fftdf-40', 'fftdf-60', 'fftdf-80', 'fftdf-100', 'fftdf-120', 'fftdf-140', 'fftdf-160', 'fftdf-180', 'fftdf-200']
    df_method = ['6', '7', '8', '9'] # '10', '15', '20', '25', '30']

    configs = [{'basis': basis, 'density-fitting-method': d} for d in df_method]
    for config in configs:
        yield config

def main(cell='diamond', method='krhf', ntasks=1, time='00:30:00', cpus_per_task=4):
    base_dir = Path(__file__).parent

    for config in loop():
        print(f"Setting up benchmark directory: {config}")

        if cell == 'diamond':
            c = config['density-fitting-method']
            config['density-fitting-method'] = "fftisdf-60-" + c
        elif cell == 'co2':
            c = config['density-fitting-method']
            config['density-fitting-method'] = "fftisdf-140-" + c
        else:
            raise ValueError(f"Cell {cell} not supported")

        dir_path = base_dir / config['density-fitting-method']
        dir_path.mkdir(parents=True, exist_ok=False)

        config['name'] = cell
        config['is-unrestricted'] = False
        config['init-guess-method'] = 'minao'

        base = Path(__file__).parent
        run_content = None

        src_path = base / '../../../src/'
        src_path = src_path.resolve()
        src_path = src_path.absolute()
        assert src_path.exists(), f"Path {src_path} not found"
        
        job_name = cell + '-' + config['density-fitting-method']

        with open(src_path / 'code/scripts/run.sh', 'r') as f:
            run_content = f.readlines()
            run_content.insert(1, f"#SBATCH --time={time}\n")
            run_content.insert(1, f"#SBATCH --mem-per-cpu=8gb\n")
            run_content.insert(1, f"#SBATCH --cpus-per-task={cpus_per_task}\n")
            run_content.insert(1, f"#SBATCH --ntasks={ntasks}\n")
            run_content.insert(1, f"#SBATCH --job-name={job_name}\n")
            # run_content.insert(1, f"#SBATCH --qos=debug\n")
            # run_content.insert(1, f"#SBATCH --constraint=icelake\n")
            run_content.insert(1, f"#SBATCH --reservation=changroup_standingres\n")

        # convert to absolute path
        python_path  = [src_path / 'fftisdf-main', src_path / 'libdmet2-main']
        python_path += [src_path / 'pyscf-forge-lnocc', src_path / 'code']

        run_content.append(f"export PYTHONPATH={python_path[0]}\n")
        for p in python_path[1:]:
            run_content.append(f"export PYTHONPATH=$PYTHONPATH:{p}\n")

        # construct the main script path, and make sure it exists
        main_path = src_path / ('code/scripts/main-%s.py' % method)
        assert main_path.exists(), f"Main script not found: {main_path}"
        run_content.append(f"cp {main_path} main.py\n")

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
    parser.add_argument("--cpus-per-task", type=int, default=4)
    args = parser.parse_args()
    kwargs = args.__dict__
    for k, v in kwargs.items():
        print(f"{k}: {v}")
    print()
    main(**kwargs)
