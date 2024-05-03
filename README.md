# Phantom-GRAPE
Phantom GRAPE is a SIMD-accelerated numerical library for N-body
simulations of self-gravitating systems. It works on x86
microprocessors with the AVX, AVX2, and AVX-512 instruction sets.

Phantom GRAPE has two sets of APIs; one is for collisionless
self-gravitating systems and compatible with that of GRAPE-5, and the
other is for collisional systems and compatible with the GRAPE-6 API.
The former API also has a capability to compute arbitrary-shaped
forces with a cutoff radius which is necessary for PPPM and TreePM
methods.

This library contains a numerical package 'quad_tree' to accelerate
the calculation of the quadrupole terms in the Barnes-Hut tree method
developed by Kodama & Ishiyama (2019) PASJ in press.  The package can
be found in the PG5 directory and be used as an extension of
Phantom-GRAPE.

For more details, see our following papers;

Tanikawa, et al. (2013) New Astronomy, 19, 74-88  (arXiv:1203:4037)

Tanikawa, et al. (2012) New Astronomy, 17, 82-92  (arXiv:1104:2700)
