source $HOME/anaconda3/bin/activate fftisdf

export PYSCF_EXT_PATH=$HOME/work/pyscf/pyscf-forge-lnocc-origin/
export PYSCF_TMPDIR=$PWD/tmp/

python -c "import pyscf; print(pyscf.__file__)"
python -c "from pyscf.pbc import lno; print(lno.__file__)"

python test.py > out.log 2>&1
