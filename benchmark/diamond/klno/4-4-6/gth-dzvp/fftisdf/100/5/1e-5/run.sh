#!/bin/bash
#SBATCH --reservation=changroup_standingres
#SBATCH --job-name=diamond-klno-4-4-6-gth-dzvp-fftisdf-100-5-1e-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=20:00:00
#SBATCH --constraint=icelake

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"

# Load environment configuration
# source /home/junjiey/anaconda3/bin/activate py38-pyscf-conda
# conda activate fftisdf-with-mkl
# source /home/junjiey/anaconda3/bin/activate fftisdf
readlink -f $HOME/anaconda3/bin/activate
source $HOME/anaconda3/bin/activate fftisdf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=1;
export OPENBLAS_NUM_THREADS=1;
export PYSCF_MAX_MEMORY=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))

echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/resnick/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
ln -s $PYSCF_TMPDIR tmp
export PYTHONPATH=/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/fftisdf-main
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/libdmet2-main
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/pyscf-forge-lnocc
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/scripts
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/scripts/main-klno.py ./main.py
python main.py --kmesh=4-4-6 --basis=gth-dzvp --lno-thresh=1e-5 --density-fitting-method=fftisdf-100-5 --name=diamond --pseudo=gth-pbe --init-guess-method=minao
echo "End time = $(date)"
