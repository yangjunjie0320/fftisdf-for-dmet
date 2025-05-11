source /Users/yangjunjie/anaconda3/bin/activate fftisdf
which python
export PYTHONPATH=$PWD/../../libdmet2-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../fftisdf-main/:$PYTHONPATH
export PYTHONPATH=$PWD/../../lno-klno/:$PYTHONPATH
export PYSCF_TMPDIR=$PWD/tmp/

python -c "import pyscf; print(pyscf.__file__)"
python -c "import libdmet; print(libdmet.__file__)"
python -c "import fft; print(fft.__file__)"

python test_patch.py

rm -rf *h5 *json 2>/dev/null
