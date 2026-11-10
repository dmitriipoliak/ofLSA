# ofLSA — matrix-free 3D global linear stability analysis in OpenFOAM

Newton solver for steady base flows and a shift-invert Arnoldi (PARPACK) eigensolver
for the leading global modes of incompressible flows, built on **OpenFOAM-14 (openfoam.org)**.

Reference: Y. Feldman et al., *Massively parallel matrix-free 3D global linear stability
analysis of incompressible flows in OpenFOAM* (in preparation).

---

## What the code does

A global linear stability analysis answers: **at which Reynolds number does a steady flow
become unstable, with what frequency, and what does the growing disturbance look like?**
It is done in two steps.

**1. Base flow — `NewtonSolver`.**
Solves the *steady* incompressible Navier–Stokes equations directly with Newton's method
(no time stepping). Each Newton step is a large linear saddle-point system for velocity and
pressure; it is solved matrix-free with FGMRES and a modified augmented-Lagrangian (mAL)
preconditioner, built from standard OpenFOAM scalar solvers. Result: steady `U`, `p`, `phi`.

**2. Eigenmodes — `StabilitySolver`.**
Linearises Navier–Stokes around that base flow. Small perturbations behave as
`exp(λ t)`, `λ = λr + iω`:

- `λr < 0` for every mode — the flow is stable; `λr > 0` for some mode — unstable;
- the critical Re is where the leading `λr` crosses zero; `ω` is the frequency of the
  oscillations that appear.

The leading eigenvalues are found by the Arnoldi method (PARPACK) in **shift-invert** mode:
you give a shift `σ` near the expected eigenvalue, and the method converges quickly to the
eigenvalues closest to it. Each Arnoldi step solves a shifted linearised system, again
matrix-free with FGMRES + mAL. Result: `λ` in the log and the mode shape as fields
`leadingEigenModeRealU/ImagU/RealP/ImagP`.

No matrix is ever assembled, so large 3D grids (256³ ≈ 17 M cells) fit in memory and run in
parallel on many cores.

---

## Quick start

```bash
source /opt/openfoam14/etc/bashrc          # or: scripts/oflsa-shell
git clone https://github.com/<owner>/<repo>.git && cd <repo>
./Allwmake                                  # builds PARPACK too if it is missing

cd tutorials/lidDrivenCavity3D
./Allrun -np 4
```

`Allrun` does: `blockMesh` → `decomposePar` → `NewtonSolver` → `StabilitySolver` →
`reconstructPar`, and prints the leading eigenvalue. Logs: `log.*`.
To run in the background: `nohup ./Allrun -np 8 > log.Allrun 2>&1 &`.

Results: `touch case.foam` and open it in ParaView; the mode is in `leadingEigenModeRealU`
(and `...ImagU`) in the last time folder.

---

## Changing the case

Copy the tutorial and edit the copy — keep the original as a reference:
`cp -r tutorials/lidDrivenCavity3D $FOAM_RUN/myCase && cd $FOAM_RUN/myCase`

| What | Where | How |
|---|---|---|
| **Reynolds number** | `system/newtonProperties` **and** `system/stabilityProperties` → `Re` | `scripts/setRe 1900` — sets both at once. They must always be equal |
| **Grid size / stretching** | `system/blockMeshDict` | `scripts/makeCavityMesh.py N a` (see below) |
| **Number of cores** | `system/decomposeParDict` | `./Allrun -np N` sets it; method `scotch` works for any N |
| **Where to look for eigenvalues** | `system/stabilityProperties` → `sigmaReal`, `sigmaImag` | put `σ` near the expected `λ`, e.g. `sigmaImag 0.58` for the cavity near Re_cr |
| **How many eigenvalues** | `stabilityProperties` → `arnoldiNev` (wanted), `arnoldiNcv` (basis size, > 2·nev) | e.g. `nev 4`, `ncv 16` |
| **Inner solve accuracy** | `stabilityProperties` → `shiftInvertTolerance` | smaller = slower but more accurate, see Troubleshooting |
| **Newton convergence** | `newtonProperties` → `newtonTolerance`, `maxNewtonIter` | |

### Grid

The cavity grid uses sine stretching towards the walls (Gelfgat, TCFD 2019):
`x_i = i/N − a·sin(2πi/N)`, the same in x, y, z.

```bash
scripts/makeCavityMesh.py 50 0.12       # tutorial grid, 125 k cells
scripts/makeCavityMesh.py 100 0.0975    # 1 M cells
scripts/makeCavityMesh.py 256 0.0975    # Gelfgat 2019 / paper grid, 16.7 M cells
scripts/makeCavityMesh.py 64 0          # uniform
```

It writes `system/blockMeshDict` in the current case (or `-o path`). Then rerun from `blockMesh`.

Rough size guide: 50³ — a workstation; 100³ — a big node; 256³ — a cluster
(the paper used 256 cores, 14–30 h).

### Re continuation

Newton converges from rest at moderate Re. At high Re start from a lower-Re solution:
compute at Re = 1000, then `scripts/setRe 1500` and run `NewtonSolver` again (without
`Allrun`, which would re-create the mesh and reset the fields): Newton writes its solution
back into the start time and starts from it next time. Repeat up to the target Re.

### Running only one step

```bash
mpirun -np 8 NewtonSolver -parallel      > log.NewtonSolver 2>&1
mpirun -np 8 StabilitySolver -parallel   > log.StabilitySolver 2>&1
```

`StabilitySolver` reads the base flow written by `NewtonSolver`, so after changing only
`sigma`/`nev` you need to rerun only the second line. After changing **Re or the grid**
rerun both.

---

## Installation details

### Option A — build from source (no root needed)

Prerequisites:

- **OpenFOAM-14** (openfoam.org): system or cluster install, or the official image via
  Apptainer (see openfoam.org/download). Ubuntu with root:
  ```bash
  sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
  sudo add-apt-repository http://dl.openfoam.org/ubuntu
  sudo apt update && sudo apt install openfoam14
  ```
- **MPI with a Fortran compiler** (`mpif90`) — on clusters `module load gcc openmpi`.
- **ARPACK/PARPACK** — found automatically:
  `ARPACK_DIR` if set → `~/software/arpack-ng-install` → system packages
  (`sudo apt install libarpack2-dev libparpack2-dev`).
  If none is found, `./Allwmake` runs `scripts/install_arpack.sh`, which builds arpack-ng
  with MPI into `$HOME` (and OpenBLAS too if there is no LAPACK). No root needed.

Binaries go to `$FOAM_USER_APPBIN`, the library to `$FOAM_USER_LIBBIN`.

### Option B — container (clusters, nothing to build)

```bash
apptainer pull oflsa.sif docker://ghcr.io/<owner>/<repo>:latest
cd mycase
apptainer exec ../oflsa.sif oflsa-run blockMesh
apptainer exec ../oflsa.sif oflsa-run decomposePar
mpirun -np 8 apptainer exec ../oflsa.sif oflsa-run NewtonSolver -parallel
mpirun -np 8 apptainer exec ../oflsa.sif oflsa-run StabilitySolver -parallel
```

Multi-node runs need the host MPI to be compatible with the container's OpenMPI.

---

## Expected results

| Case | Re_cr | ω |
|---|---|---|
| Cubic lid-driven cavity | ≈ 1928.75 | ≈ 0.584 |
| Diagonally driven cavity | ≈ 2320.8 | ≈ 0.251 |

Values for the 256³ grid. The 50³ tutorial is a quick check of the whole chain.

---

## Repository layout

| Path | What |
|---|---|
| `src/matrixFreeFGMRESSafe` | Matrix-free FGMRES library |
| `applications/solvers/NewtonSolver` | Steady base flow (Newton + FGMRES + mAL) |
| `applications/solvers/StabilitySolver` | Leading eigenmodes (shift-invert + PARPACK) |
| `tutorials/lidDrivenCavity3D` | Test case |
| `etc/bashrc` | Environment, locates ARPACK |
| `scripts/oflsa-shell` | Clean shell with OpenFOAM-14 + ofLSA loaded |
| `scripts/install_arpack.sh` | Builds arpack-ng (+ OpenBLAS) in `$HOME` |
| `scripts/setRe` | Sets Re in both solver dictionaries |
| `scripts/makeCavityMesh.py` | Generates the stretched cavity grid |
| `container/` | Docker/Apptainer image |

---

## Troubleshooting

- **`source .../openfoam14/etc/bashrc` hangs**, or `_foamSource: command not found`
  (seen on Ubuntu 26.04 desktops; OpenFOAM calls `paraview --version`, which hangs) —
  use `scripts/oflsa-shell`, or press Ctrl+C once: only the ParaView check is interrupted.
- **`exe-not-found` from mpirun** — the OpenFOAM environment is not loaded in that shell.
- **StabilitySolver stops: `PARPACK Arnoldi basis vector ... left the required discrete
  homogeneous solenoidal space`** — the divergence of the Arnoldi vectors grew above
  `arnoldiConstraintTolerance` because the inner solves are inexact. Tighten
  `shiftInvertTolerance` (e.g. `1e-6`) and/or relax `arnoldiConstraintTolerance` (e.g. `1e-3`).
- **`cannot find -liomp5 / -lifport ...`** — an old `Make/options` with the Intel Fortran
  runtime; use the ones in this repository (gfortran).
- **ParaView does not open from a terminal** — `nohup paraview > /dev/null 2>&1 < /dev/null &`.

## License

GPL-3.0-or-later (as OpenFOAM).
