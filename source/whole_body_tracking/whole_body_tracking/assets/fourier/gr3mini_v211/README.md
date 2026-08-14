# Fourier GR3Mini V2.1.1 assets

The MJCF, scene files, textures, and STL meshes in this directory were copied
from `any2track/storage/assets/fourier_gr3mini_v211`.

`urdf/gr3mini_v211.urdf` is the Isaac Lab model for the same V2.1.1 robot. Its
mesh paths were adjusted to reuse the copied `assets/` directory, and its two
head joints are fixed to match UFO's `gr3mini_v211_23dof` training model. The
original 25-DoF MJCF remains alongside it for sim-to-sim comparison.
