# FFT-ISDF for DMET

## Overview

This is **fftisdf-for-dmet**, a quantum chemistry project that combines FFT-ISDF methods with multiple embedding theory methods for periodic systems. The project integrates multiple specialized computational chemistry libraries to perform electronic structure calculations on crystalline materials.

## How to run

Clone the repository with submodules
```bash
git clone --recurse-submodules https://github.com/yangjunjie0320/fftisdf-for-dmet.git
```
Note that `libdmet2` and `lno-klno` are not yet public, please verify if you have 
the correct access to them.

We recommend using conda to manage the environment. The environment file is in `src/fftisdf-main/environment.yml`:
```bash
# Create environment from submodule
conda env create --file=src/fftisdf-main/environment.yml --name=fftisdf
conda activate fftisdf

# Add all source directories to PYTHONPATH
export PYTHONPATH=$PWD/src/fftisdf-main:$PYTHONPATH
export PYTHONPATH=$PWD/src/libdmet2-main:$PYTHONPATH
export PYTHONPATH=$PWD/src/fcdmft-main:$PYTHONPATH
export PYTHONPATH=$PWD/src/pyscf-forge-lnocc:$PYTHONPATH
export PYTHONPATH=$PWD/src/code:$PYTHONPATH
```
