source /Users/yangjunjie/anaconda3/bin/activate fftisdf

export PYTHONPATH=$PWD/../../src/libdmet2-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/fftisdf-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/lno-klno/:$PYTHONPATH
export PYTHONPATH=$PWD/../../src/scripts/:$PYTHONPATH

export PYSCF_TMPDIR=$PWD/tmp/

python -c "import pyscf; print(pyscf.__file__)"
python -c "import libdmet; print(libdmet.__file__)"
python -c "import fft; print(fft.__file__)"
python -c "import lno.base; print(lno.base.__file__)"

python test.py

rm -rf *h5 *json