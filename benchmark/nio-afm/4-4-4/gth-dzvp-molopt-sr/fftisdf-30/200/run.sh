#!/bin/bash
#SBATCH --reservation=changroup_standingres
#SBATCH --job-name=nio-afm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=04:00:00
#SBATCH --constraint=icelake

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"

# Load environment configuration
# source /home/junjiey/anaconda3/bin/activate py38-pyscf-conda
source /home/junjiey/anaconda3/bin/activate fftisdf-with-mpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=1;

export PYSCF_MAX_MEMORY=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/resnick/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
ln -s $PYSCF_TMPDIR tmp
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-with-dmet/benchmark/nio-afm/../../src/fftisdf-main:/resnick/groups/changroup/members/junjiey/fftisdf-with-dmet/benchmark/nio-afm/../../src/libdmet2-main:/resnick/groups/changroup/members/junjiey/fftisdf-with-dmet/benchmark/nio-afm/../../src/scripts
cp /resnick/groups/changroup/members/junjiey/fftisdf-with-dmet/benchmark/nio-afm/../../src/scripts/main-kuhf.py ./main.py
python main.py --kmesh=4-4-4 --basis=gth-dzvp-molopt-sr --density-fitting-method=fftisdf-30 --ke-cutoff=200 --name=nio-afm --pseudo=gth-pbe --init-guess-method=minao --df-to-read=None --is-unrestricted
echo "End time = $(date)"
