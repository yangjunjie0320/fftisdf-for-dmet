#!/bin/bash
#SBATCH --job-name=nio-afm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=00:30:00
#SBATCH --constraint=icelake

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"
source /home/junjiey/anaconda3/bin/activate fftisdf-with-mkl

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export PYSCF_MAX_MEMORY=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))

echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export PYTHONPATH=$PWD/../../src/libdmet2-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/fftisdf-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/lno-klno/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/scripts/:$PYTHONPATH

export TMP=/resnick/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
ln -s $PYSCF_TMPDIR tmp

export PYTHONPATH=$PWD/../../src/libdmet2-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/fftisdf-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/lno-klno/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/scripts/:$PYTHONPATH

python test.py

echo "End time = $(date)"
