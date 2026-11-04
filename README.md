# ofLSA — matrix-free 3D global linear stability analysis in OpenFOAM

Newton solver for steady base flows and a shift-invert Arnoldi (PARPACK) eigensolver
for the leading global modes of incompressible flows, built on OpenFOAM-14 (openfoam.org).

Reference: Y. Feldman et al., *Massively parallel matrix-free 3D global linear stability
analysis of incompressible flows in OpenFOAM* (in preparation).

## Contents

| Path | What |
|---|---|
| `src/matrixFreeFGMRESSafe` | Matrix-free FGMRES library |
| `applications/solvers/NewtonSolver` | Steady base flow by Newton iterations |
| `applications/solvers/StabilitySolver` | Leading eigenmodes, shift-invert + PARPACK |
| `tutorials/lidDrivenCavity3D` | Test case: 3D lid-driven cavity |
| `etc/bashrc` | Environment (locates ARPACK) |
| `scripts/install_arpack.sh` | Builds arpack-ng with MPI in `$HOME`, no root |
| `container/` | Ready-made image: OpenFOAM-14 + PARPACK + solvers |

Everything runs from the command line; no GUI is needed. Results open in ParaView
on any machine, or are post-processed headless with `foamToVTK` / `pvpython`.

---

## Option A — container (recommended on clusters, no root, nothing to build)

Requires Apptainer (Singularity), available on most HPC systems.

```bash
apptainer pull oflsa.sif docker://ghcr.io/<owner>/<repo>:latest

cp -r <repo>/tutorials/lidDrivenCavity3D mycase && cd mycase
apptainer exec ../oflsa.sif oflsa-run blockMesh
apptainer exec ../oflsa.sif oflsa-run decomposePar
mpirun -np 8 apptainer exec ../oflsa.sif oflsa-run NewtonSolver -parallel
mpirun -np 8 apptainer exec ../oflsa.sif oflsa-run StabilitySolver -parallel
```

On a single node you can also run MPI inside the container:
`apptainer exec oflsa.sif oflsa-run mpirun -np 8 StabilitySolver -parallel`.
For multi-node runs the host MPI must be compatible with the container's OpenMPI
(hybrid model); ask your cluster support which OpenMPI module to load.

## Option B — build from source (no root)

### 1. Prerequisites

- **OpenFOAM-14** (openfoam.org), any of:
  - system or cluster install: `source /opt/openfoam14/etc/bashrc` or `module load ...`
  - official image via Apptainer (see openfoam.org/download)
  - build from source (slow, 1–3 h)
- **MPI** with a Fortran compiler (`mpif90`), **BLAS/LAPACK**, **cmake** — on clusters
  usually `module load gcc openmpi cmake openblas`
- **ARPACK/PARPACK** — one of:
  - already available: `export ARPACK_DIR=/path/to/arpack-ng`
  - Debian/Ubuntu with root: `sudo apt install libarpack2-dev libparpack2-dev`
  - no root: `scripts/install_arpack.sh` (≈5 min, installs to `$HOME/software/arpack-ng-install`)

### 2. Build

```bash
source /opt/openfoam14/etc/bashrc     # or your module
git clone https://github.com/<owner>/<repo>.git
cd <repo>
scripts/install_arpack.sh             # only if PARPACK is not available
source etc/bashrc
./Allwmake -j
```

Binaries go to `$FOAM_USER_APPBIN`, the library to `$FOAM_USER_LIBBIN` — inside `$HOME`.

### 3. Run the test case

```bash
cp -r tutorials/lidDrivenCavity3D $FOAM_RUN/
cd $FOAM_RUN/lidDrivenCavity3D
./Allrun -np 8
```

On a scheduler (SGE/Slurm) put the same commands into a job script.

## Expected results

| Case | Re_cr | ω |
|---|---|---|
| Cubic lid-driven cavity | ≈ 1928.75 | ≈ 0.584 |
| Diagonally driven cavity | ≈ 2320.8 | ≈ 0.251 |

Reference values are for the 256³ grid (256 cores, 14–30 h). The tutorial grid is coarse
and only checks that the chain works.

## Troubleshooting

- `source .../openfoam14/etc/bashrc` hangs, or prints `_foamSource: command not found`
  (seen on Ubuntu 26.04 desktops) — use a clean shell instead:
  `scripts/oflsa-shell` (loads OpenFOAM and ofLSA in one step), or manually
  `env -i HOME=$HOME USER=$USER TERM=$TERM DISPLAY=$DISPLAY bash --norc` and then
  `source /opt/openfoam14/etc/bashrc < /dev/null`.
- StabilitySolver stops with `PARPACK Arnoldi basis vector ... left the required discrete
  homogeneous solenoidal space` — divergence of the Arnoldi basis grew above
  `arnoldiConstraintTolerance` because of inexact inner solves. Tighten
  `shiftInvertTolerance` (e.g. 1e-6) and/or relax `arnoldiConstraintTolerance`
  in `system/stabilityProperties`.
- `prun:exe-not-found` from mpirun — the OpenFOAM environment is not loaded in that shell.

- `cannot find -liomp5 / -lifport ...` — an old `Make/options` with Intel Fortran runtime;
  use the one in this repository (gfortran).
- `PARPACK not found` — run `scripts/install_arpack.sh` or set `ARPACK_DIR`, then `source etc/bashrc` again.
- ParaView hangs on start in a Wayland session — launch detached:
  `nohup paraview > /dev/null 2>&1 < /dev/null &`.

## License

GPL-3.0-or-later (as OpenFOAM).
