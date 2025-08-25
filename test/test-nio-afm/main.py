import numpy
from pyscf.pbc import gto, scf, dft
from pyscf.df import aug_etb

alph_label = ["0 Ni 3dx2-y2", "0 Ni 3dz\^2 "]
beta_label = ["1 Ni 3dx2-y2", "1 Ni 3dz\^2 "]

# from libdmet.utils.iotools import read_poscar
cell = gto.Cell()
cell.atom = """
Ni 0.0000 0.0000 0.0000
Ni 4.1700 4.1700 4.1700
O  2.0850 2.0850 2.0850
O  6.2550 6.2550 6.2550
"""
cell.a = """
4.1700 2.0850 2.0850
2.0850 4.1700 2.0850
2.0850 2.0850 4.1700
"""
cell.unit = "A"
cell.basis = 'gth-dzvp-molopt-sr'
cell.pseudo = "gth-hf-rev"
cell.ke_cutoff = 200
cell.exp_to_discard = 0.1
cell.verbose = 5
cell.build()

alph_index = cell.search_ao_label(alph_label)
beta_index = cell.search_ao_label(beta_label)
print("alph_label = %s, alph_index = %s" % (alph_label, alph_index))
print("beta_label = %s, beta_index = %s" % (beta_label, beta_index))

nao = cell.nao_nr()
kmesh = [1, 1, 1]
kpts = cell.make_kpts(kmesh)
nkpt = len(kpts)

# method 1: with GDF
mf = scf.KUHF(cell, kpts=kpts).density_fit()
mf.exxdiv = "ewald"
mf.conv_tol = 1e-6

dm0 = mf.get_init_guess(key='minao')
assert dm0.shape == (2, nkpt, nao, nao)

dm0[0, :, alph_index, alph_index] *= 1.0
dm0[0, :, beta_index, beta_index] *= 0.0
dm0[1, :, alph_index, alph_index] *= 0.0
dm0[1, :, beta_index, beta_index] *= 1.0

mf.with_df.auxbasis = aug_etb(cell, beta=1.2)
mf.with_df.build()
mf.kernel(dm0)

# ene_hf_gdf   = -366.72991923
print("ene_hf_gdf   = %12.8f" % mf.e_tot)

dm0 = mf.make_rdm1()

# method 2: with FFTDF
mf = scf.KUHF(cell, kpts=kpts)
mf.exxdiv = "ewald"
mf.conv_tol = 1e-6
mf.kernel(dm0)

# ene_hf_fftdf = -366.72992947
print("ene_hf_fftdf = %12.8f" % mf.e_tot)
