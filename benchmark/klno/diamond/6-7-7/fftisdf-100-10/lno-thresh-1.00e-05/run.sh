#!/bin/bash
#SBATCH --job-name=diamond-fftisdf-100-10-kmesh-6-7-7-lno-thresh-1.00e-05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=04:00:00

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
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/fcdmft-main
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/pyscf-forge-lnocc
export PYTHONPATH=$PYTHONPATH:/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code

cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/src/code/scripts/main-klno.py main.py
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/benchmark/krhf-dmet/diamond/6-7-7/fftisdf-100-10/scf.chk scf.chk
cp /resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/benchmark/krhf-dmet/diamond/6-7-7/fftisdf-100-10/tmp/df.h5 tmp/df.h5

python main.py --basis=cc-pvdz --pseudo=gth-hf-rev --kmesh=6-7-7 --density-fitting-method=fftisdf-100-10 --lno-thresh=1e-05 --name=diamond --init-guess-method=chk --df-to-read=./tmp/df.h5 --kconserv-to-read=/resnick/groups/changroup/members/junjiey/fftisdf-for-dmet/test/test-8-8-10/diamond-kconserv-wrap-around-1.chk
echo "End time = $(date)"
