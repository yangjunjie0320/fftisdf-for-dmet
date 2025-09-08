import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path

def loop(cell='diamond'):
    basis = 'cc-pvdz'
    pseudo = 'gth-hf-rev'

    df_method = []
    # df_method = ['gdf-2.0', 'rsdf-2.0']
    
    if cell == 'diamond':
        df_method += ['fftdf-60', 'fftdf-80', 'fftdf-100', 'fftdf-120', 'fftdf-140', 'fftdf-160', 'fftdf-180', 'fftdf-200']
        df_method += ['fftisdf-60-8', 'fftisdf-60-10', 'fftisdf-60-12', 'fftisdf-60-14', 'fftisdf-60-16', 'fftisdf-60-18', 'fftisdf-60-20']
        df_method += ['fftisdf-80-8', 'fftisdf-80-10', 'fftisdf-80-12', 'fftisdf-80-14', 'fftisdf-80-16', 'fftisdf-80-18', 'fftisdf-80-20']
        df_method += ['fftisdf-100-8', 'fftisdf-100-10', 'fftisdf-100-12', 'fftisdf-100-14', 'fftisdf-100-16', 'fftisdf-100-18', 'fftisdf-100-20']
    
    elif cell == 'co2':
        df_method += ['fftisdf-140-14']
        # df_method += ['fftdf-140', 'fftdf-160', 'fftdf-180', 'fftdf-200']
        # df_method += ['fftisdf-120-8', 'fftisdf-120-10', 'fftisdf-120-12', 'fftisdf-120-14', 'fftisdf-120-16', 'fftisdf-120-18', 'fftisdf-120-20']
        # df_method += ['fftisdf-140-8', 'fftisdf-140-10', 'fftisdf-140-12', 'fftisdf-140-14', 'fftisdf-140-16', 'fftisdf-140-18', 'fftisdf-140-20']
        # df_method += ['fftisdf-160-8', 'fftisdf-160-10', 'fftisdf-160-12', 'fftisdf-160-14', 'fftisdf-160-16', 'fftisdf-160-18', 'fftisdf-160-20']
        # df_method += ['fftisdf-180-8', 'fftisdf-180-10', 'fftisdf-180-12', 'fftisdf-180-14', 'fftisdf-180-16', 'fftisdf-180-18', 'fftisdf-180-20']

    kmesh  = ['1-1-2', '1-2-2', '2-2-2']
    kmesh += ['2-2-3', '2-3-3', '3-3-3']
    # kmesh += ['3-3-4', '3-4-4', '4-4-4']

    from itertools import product
    configs = [{'basis': basis, 'pseudo': pseudo, 'kmesh': k, 'density-fitting-method': d} for k, d in product(kmesh, df_method)]
    for config in configs:
        yield config

def main(cell='diamond', method='krhf', ntasks=1, time='00:30:00', cpus_per_task=4, reservation=None):
    base_dir = Path(__file__).parent

    for config in loop(cell=cell):
        config['density-fitting-method'] = config['density-fitting-method']

        print(f"Setting up benchmark directory: {config}")
        dir_path = base_dir / config['kmesh'] / config['density-fitting-method'] 
        # dir_path = dir_path / ("lno-thresh-%6.2e" % config['lno-thresh'])
        if dir_path.exists():
            print(f"Directory {dir_path} already exists, deleting")
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=False)

        ref_path = base_dir / ".." / ".." / 'krhf-dmet' / cell / config['kmesh'] / config['density-fitting-method']
        ref_path = ref_path.resolve()
        ref_path = ref_path.absolute()
        assert ref_path.exists(), f"Reference path {ref_path} not found"

        config['name'] = cell
        config['is-unrestricted'] = False
        config['init-guess-method'] = 'chk'
        # config['lno-thresh'] = config['lno-thresh']
        config['df-to-read'] = './tmp/df.h5'
        # config['kconserv-to-read'] = "/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/test/test-8-8-10/diamond-kconserv-wrap-around-1.chk"

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
            run_content.insert(1, f"#SBATCH --mem-per-cpu=10gb\n")
            run_content.insert(1, f"#SBATCH --cpus-per-task={cpus_per_task}\n")
            run_content.insert(1, f"#SBATCH --ntasks={ntasks}\n")
            run_content.insert(1, f"#SBATCH --job-name={job_name}\n")
            if reservation:
                run_content.insert(1, f"#SBATCH --reservation={reservation}\n")

                if "h100" in reservation:
                    run_content.insert(1, f"#SBATCH --partition=gpu\n")

        # convert to absolute path
        python_path  = [src_path / 'fftisdf-main', src_path / 'libdmet2-main']
        python_path += [src_path / 'fcdmft-main']
        python_path += [src_path / 'pyscf-forge-lnocc', src_path / 'code']

        run_content.append(f"export PYTHONPATH={python_path[0]}\n")
        for p in python_path[1:]:
            run_content.append(f"export PYTHONPATH=$PYTHONPATH:{p}\n")

        # construct the main script path, and make sure it exists
        main_path = src_path / ('code/scripts/main-%s.py' % method)
        assert main_path.exists(), f"Main script not found: {main_path}"

        run_content.append(f"\ncp {main_path} main.py\n")
        run_content.append(f"cp {ref_path / 'scf.chk'} scf.chk\n")
        run_content.append(f"cp {ref_path / 'tmp' / 'df.h5'} tmp/df.h5\n\n")

        is_unrestricted = config.pop('is-unrestricted')
        cmd = "python main.py %s" % " ".join([f"--{k}={v}" for k, v in config.items()])
        if is_unrestricted:
            cmd += " --is-unrestricted"
        run_content.append(cmd + "\n")

        run_content.append(f"\nrm tmp; rm -rf $PYSCF_TMPDIR\n")
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
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--time", type=str, default="00:30:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)

    # reservation is a string, default to None
    parser.add_argument("--reservation", type=str, default="changroup_standingres")
    args = parser.parse_args()
    kwargs = args.__dict__

    pwd = Path(__file__).parent
    kwargs['cell'] = pwd.name
    kwargs['method'] = pwd.parent.name

    for k, v in kwargs.items():
        print(f"{k}: {v}")
    print()
    main(**kwargs)
