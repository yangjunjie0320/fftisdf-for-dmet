import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path
import glob

def loop(cell='diamond'):
    basis = 'cc-pvdz'
    pseudo = 'gth-hf-rev'

    df_method = []
    # df_method.append('gdf-1.2')
    # df_method.append('gdf-1.4')
    # df_method.append('gdf-1.6')
    # df_method.append('gdf-1.8')
    df_method.append('gdf-2.0')
    
    assert cell == "nio-afm"
    df_method += ["fftisdf-180-25"]

    kmesh = []
    kmesh += ['1-1-2', '1-2-2', '2-2-2']
    kmesh += ['2-2-3', '2-3-3', '3-3-3']
    kmesh += ['3-3-4', '3-4-4', '4-4-4']
    kmesh += ['4-4-6'] # , '4-6-6', '6-6-6']
    # kmesh += ['6-6-7', '6-7-7', '7-7-7']
    # kmesh += ['7-7-8', '7-8-8', '8-8-8']
    # kmesh += ['8-8-10', '8-10-10', '10-10-10']

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
        assert dir_path.exists(), f"Directory {dir_path} not found"
        print(f"Directory {dir_path} found")
        is_out_log_exist = os.path.exists(dir_path / 'out.log')
        assert is_out_log_exist

        # search for slurm log
        import glob
        slurm_log_list = glob.glob(str(dir_path / 'slurm-*'))
        assert len(slurm_log_list) == 1
        slurm_log_file = slurm_log_list[0]
        print(f"slurm_log_files: {slurm_log_file}")
        
        lines = None
        with open(slurm_log_file, 'r') as f:
            lines = f.read()
        is_dmet_converged = "DMET converged after" in lines
        if is_dmet_converged:
            print(f"DMET converged, skipping {dir_path}")
            continue

        # clean up the directory
        files_to_be_removed = dir_path.glob(str(dir_path / '*.h5'))
        files_to_be_removed += dir_path.glob(str(dir_path / '*.json'))
        files_to_be_removed += dir_path.glob(str(dir_path / 'slurm-*'))
        for f in files_to_be_removed:
            print(f"Removing {f}")
            os.remove(f)

        os.chdir(dir_path)
        tmp_real_path = os.path.realpath('tmp')
        df_h5_path = os.path.join(tmp_real_path, 'df.h5')
        assert os.path.exists(df_h5_path)
        os.system("rm tmp")

        # if dir_path.exists():
        #     print(f"Directory {dir_path} already exists, deleting")
        #     shutil.rmtree(dir_path)
        # dir_path.mkdir(parents=True, exist_ok=False)

        # ref_path = base_dir / ".." / ".." / 'kuhf-dmet' / cell / config['kmesh'] / 'fftisdf-180-24'
        # ref_path = ref_path.resolve()
        # ref_path = ref_path.absolute()
        # assert ref_path.exists(), f"Reference path {ref_path} not found"

        config['name'] = cell
        config['is-unrestricted'] = ("afm" in cell.lower())
        config['init-guess-method'] = 'chk'
        config['df-to-read'] = df_h5_path

        base = Path(__file__).parent
        run_content = None

        src_path = base / '../../../src/'
        src_path = src_path.resolve()
        src_path = src_path.absolute()
        assert src_path.exists(), f"Path {src_path} not found"
        
        job_name = cell + '-' + config['density-fitting-method']
        job_name += '-' + 'kmesh-' + config['kmesh']

        with open(src_path / 'code/scripts/run.sh', 'r') as f:
            run_content = f.readlines()
            run_content.insert(1, f"#SBATCH --time={time}\n")
            run_content.insert(1, f"#SBATCH --mem-per-cpu=6gb\n")
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
        # run_content.append(f"cp {ref_path / 'scf.chk'} scf.chk\n")
        # run_content.append(f"cp {ref_path / 'tmp' / 'df.h5'} tmp/df.h5\n\n")

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
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--time", type=str, default="00:30:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
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
