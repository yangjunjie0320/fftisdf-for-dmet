#!/bin/bash
#SBATCH --reservation=changroup_standingres
#SBATCH --qos=debug
#SBATCH --job-name=diamond-fftisdf-60-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=00:05:00

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
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code

kmesh=4-4-4
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code/scripts/main-krpa.py main.py
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code/scripts/main-krpa.py main.py

rm out.log isdf.h5
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/benchmark/krpa/diamond/${kmesh}/fftisdf-60-8/main.py .
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/benchmark/krpa/diamond/${kmesh}/fftisdf-60-8/tmp/isdf.h5 .
# cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code/krpa.py .

LINE_PROFILE=1 python main.py --basis=cc-pvdz --density-fitting-method=fftisdf-60-8 \
       --kmesh=$kmesh --df-to-read=isdf.h5 \
       --name=diamond --init-guess-method=minao      

echo "End time = $(date)"
