# ofLSA — global linear stability analysis of 3D flows in OpenFOAM

*Русская версия: [README.ru.md](README.ru.md)*

This repository answers one question about a flow: **at what Reynolds number does a steady
flow stop being steady, and what does the disturbance that grows look like?**

It contains two solvers for OpenFOAM-14 (openfoam.org), written by Y. Feldman's group
(Ben-Gurion University), and a ready-to-run example: the 3D lid-driven cavity.

---

## 1. The idea in five minutes

Take a cube filled with fluid whose top lid slides at constant speed. At low Re the fluid
settles into a steady vortex. Raise Re, and at some point the steady vortex can no longer
survive: any tiny disturbance grows and the flow starts to oscillate. That point is the
**critical Reynolds number**. For the cube it is Re ≈ 1929.

We find it in two steps — one solver each.

**Step 1 — the steady flow (`NewtonSolver`).**
We need the steady flow even where it is unstable (nature would never show it to us there,
but mathematically it exists). A time-stepping solver cannot find it — it would drift away
into the oscillating flow. So we solve the *steady* Navier–Stokes equations directly with
Newton's method: guess, correct, repeat, until the residual is tiny.

**Step 2 — will a small disturbance grow? (`StabilitySolver`).**
We add a tiny disturbance to the steady flow and ask how it evolves. Because it is tiny,
its equations are linear, and their solutions behave like `e^(λt)` with a complex number
`λ = λr + i·ω`:

| | meaning |
|---|---|
| `λr < 0` | the disturbance dies out — the flow is **stable** |
| `λr > 0` | it grows — the flow is **unstable** |
| `λr = 0` | the boundary: this Re is the **critical** one |
| `ω` | the angular frequency of the oscillations that will appear |

The solver finds the few `λ` with the largest `λr` (the most dangerous disturbances) and
writes out their shape — the **eigenmode** — which you can look at in ParaView.

**Why it is hard, and what is special here.** A 256³ grid has about 17 million cells, i.e.
~70 million unknowns. Building and inverting such a matrix is impossible. Here no matrix is
ever built: every operation is done "on the fly" with OpenFOAM's own operators, and the
big linear systems are solved iteratively (FGMRES) with a good preconditioner (modified
augmented Lagrangian, mAL). The eigenvalues are found with PARPACK (Arnoldi method) in
*shift-invert* mode: you tell it roughly where to look (`σ`), and it quickly finds the
eigenvalues nearest to that point.

---

## 2. What you need

- **Linux** (a workstation or a cluster; no root needed).
- **OpenFOAM-14** from openfoam.org. *Not* the ESI version (openfoam.com, v2406/v2606) —
  that is a different code base.
- **MPI** (OpenMPI) with a Fortran compiler — usually comes with OpenFOAM; on a cluster
  `module load gcc openmpi`.
- **ARPACK/PARPACK** — if you don't have it, the build script installs it into your home
  directory automatically.

Installing OpenFOAM-14 on Ubuntu (needs root):
```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt install openfoam14 libarpack2-dev libparpack2-dev
```
On a cluster without root: use the cluster's OpenFOAM-14 module, or the container (section 8).

---

## 3. First run, step by step

```bash
git clone https://github.com/<owner>/<repo>.git ofLSA
cd ofLSA
```

**Load the environment.**
```bash
scripts/oflsa-shell
```
This opens a new shell with OpenFOAM-14 and this project loaded; the prompt becomes
`(ofLSA) ...$`. Type the next commands *after* the prompt appears.
(Alternative: `source /opt/openfoam14/etc/bashrc`, then `source etc/bashrc`.)

**Build.**
```bash
./Allwmake
```
Compiles the FGMRES library and both solvers (a couple of minutes). If PARPACK is not
found, it is built first (5–20 min). At the end you should see:
```
Built:
.../bin/NewtonSolver
.../bin/StabilitySolver
```

**Run the example.**
```bash
cd tutorials/lidDrivenCavity3D
nohup ./Allrun -np 4 > log.Allrun 2>&1 &
tail -f log.Allrun
```
- `-np 4` — number of CPU cores to use. Use as many as you have.
- `nohup ... &` — runs in the background, keeps running if you close the terminal.
- `tail -f` — shows progress; `Ctrl+C` leaves the view, the run continues.

What `Allrun` does, in order:

1. `blockMesh` — builds the grid (50×50×50, refined towards the walls);
2. `decomposePar` — cuts it into pieces, one per core;
3. `NewtonSolver` — the steady flow, raising Re gradually (200 → 500 → 1000 → 1250 → 1500 → … → target),
   because Newton does not converge if you start from rest at high Re;
4. `StabilitySolver` — the eigenvalues and eigenmode at the target Re;
5. `reconstructPar` — glues the pieces back together.

Expect minutes for the Newton part and longer for the stability part, depending on cores.

---

## 4. Reading the results

**In `log.Allrun`** — one line per Newton step:
```
NewtonSolver at Re = 1000 ...
Newton 6: momentum=3.1e-08 continuity=2.2e-12
```
Small numbers = converged.

**The answer** — at the end of `log.Allrun` (or `log.StabilitySolver`):
```
growth rate Re(lambda)      = -0.0123      <- λr: negative = stable
angular frequency Im(lambda)= 0.581        <- ω
```

**The eigenmode in ParaView:**
```bash
paraview case.foam
```
Click *Apply*, pick the field `leadingEigenModeRealU` (and `leadingEigenModeImagU`).
These are the real and imaginary parts of the growing disturbance; an iso-surface of one
velocity component shows its 3D structure.

---

## 5. What is in the case folder

```
tutorials/lidDrivenCavity3D/
├── 0/U, 0/p                  initial and boundary conditions (lid velocity = 1)
├── system/
│   ├── blockMeshDict         the grid
│   ├── newtonProperties      settings of the steady solver   (Re, tolerances)
│   ├── stabilityProperties   settings of the eigen solver    (Re, shift σ, number of modes)
│   ├── decomposeParDict      how to split over cores (set by Allrun)
│   ├── controlDict, fvSchemes, fvSolution    standard OpenFOAM files
├── Allrun                    runs everything
└── Allclean                  deletes results, keeps settings
```

Re is defined with the lid speed (1) and the cube side (1): ν = 1/Re.

---

## 6. Recipes: changing the case

Work on a copy, keep the original:
```bash
cp -r tutorials/lidDrivenCavity3D ~/myCavity && cd ~/myCavity
```

The helper scripts (`setRe`, `makeCavityMesh.py`) are on the PATH once the environment is
loaded, so you can call them from any case folder.

**Change the Reynolds number**
```bash
setRe 1900
./Allrun -np 8
```
`setRe` changes Re in *both* `newtonProperties` and `stabilityProperties`. They must always
be equal — the stability is computed for the flow at that Re.

**Change the grid**
```bash
makeCavityMesh.py 100 0.0975          # run inside the case folder
```
`100` — cells in each direction; `0.0975` — how strongly cells are squeezed towards the
walls (0 = uniform). It overwrites `system/blockMeshDict`; then run `./Allrun` again. Grid sizes: 50³ — any PC; 100³ (1 M cells) — a strong workstation;
256³ (17 M) — a cluster (the paper used 256 cores and 14–30 hours).

**Use more or fewer cores** — just `./Allrun -np 16`.

**Look for other eigenvalues** — in `system/stabilityProperties`:
- `sigmaImag 0.5` — where to look along the frequency axis; put it near the `ω` you expect;
- `arnoldiNev 2` — how many eigenvalues to find; `arnoldiNcv` — the working basis,
  keep it at least `2·nev + 1`.

After changing only these, you do not need to recompute the steady flow:
```bash
mpirun -np 8 StabilitySolver -parallel > log.StabilitySolver 2>&1
```

**Newton fails at some Re** — `Allrun` halves the Re step by itself and retries
(down to a step of 10). If it still stops, the last converged base flow stays on disk and its
Re is in the file `lastConvergedRe`; continue from it without rebuilding the mesh:
```bash
./Allrun -np 8 -continue                 # from lastConvergedRe
./Allrun -np 8 -continue -from 1500      # if you know the Re of the flow on disk
```

**Find the critical Re** — run at a few Re around the expected value (e.g. 1850, 1900,
1950, 2000), note `λr` for each, and interpolate to where `λr = 0`.

---

## 7. When something goes wrong

| What you see | Why | What to do |
|---|---|---|
| `source .../bashrc` hangs, or `_foamSource: command not found` | OpenFOAM calls `paraview --version` at load time; on some Ubuntu desktops that hangs | use `scripts/oflsa-shell`; or press `Ctrl+C` once |
| `exe-not-found` from `mpirun` | OpenFOAM is not loaded in this terminal | `scripts/oflsa-shell` first |
| `Newton FGMRES failed after 200 iterations` | Newton started too far from the solution | `./Allrun -continue -ramp "..."` with smaller steps |
| `PARPACK Arnoldi basis vector ... left the required ... solenoidal space` | small errors of the inner solves accumulated | in `stabilityProperties`: `shiftInvertTolerance 1e-6` and/or `arnoldiConstraintTolerance 1e-3` |
| `cannot find -liomp5 / -lifport` | an old `Make/options` with the Intel compiler | use the files from this repository |
| `PARPACK not found` | ARPACK not installed | `scripts/install_arpack.sh`, or `sudo apt install libarpack2-dev libparpack2-dev` |
| ParaView does not open from a terminal | same ParaView issue | `nohup paraview case.foam > /dev/null 2>&1 < /dev/null &` |

---

## 8. Running on a cluster without OpenFOAM: container

The container has OpenFOAM-14, PARPACK and the solvers inside. It needs Apptainer
(Singularity), which most clusters have:
```bash
apptainer pull oflsa.sif docker://ghcr.io/<owner>/<repo>:latest
cd ~/myCavity
apptainer exec ~/oflsa.sif oflsa-run blockMesh
apptainer exec ~/oflsa.sif oflsa-run decomposePar
mpirun -np 32 apptainer exec ~/oflsa.sif oflsa-run NewtonSolver -parallel
mpirun -np 32 apptainer exec ~/oflsa.sif oflsa-run StabilitySolver -parallel
```
For runs over several nodes the cluster's OpenMPI must be compatible with the one in the
container — ask your cluster support which module to load.

---

## 9. Reference values

| Case | Re_cr | ω |
|---|---|---|
| Cubic lid-driven cavity | ≈ 1928.75 | ≈ 0.584 |
| Diagonally driven cavity | ≈ 2320.8 | ≈ 0.251 |

On the 256³ grid. The 50³ example is coarse — it checks that everything works, the numbers
will differ somewhat.

---

## Reproducing the paper

`tutorials/paper/` contains the two 256³ cases of the paper (straight and diagonally driven
cavity) with the paper's Re, grid and shift. They need a cluster; see
[tutorials/paper/README.md](tutorials/paper/README.md).

## 10. Repository layout

| Path | What it is |
|---|---|
| `src/matrixFreeFGMRESSafe` | iterative linear solver (FGMRES) without assembled matrices |
| `applications/solvers/NewtonSolver` | steady flow by Newton's method |
| `applications/solvers/StabilitySolver` | eigenvalues and eigenmodes (PARPACK, shift-invert) |
| `tutorials/lidDrivenCavity3D` | the example (50³, quick check) |
| `tutorials/paper` | the two 256³ cases of the paper |
| `scripts/oflsa-shell` | opens a terminal with everything loaded |
| `scripts/setRe` | sets Re in both settings files |
| `scripts/makeCavityMesh.py` | generates the cavity grid |
| `scripts/install_arpack.sh` | installs PARPACK into your home directory |
| `etc/bashrc` | finds PARPACK, sets paths |
| `container/` | recipe of the container |

## Citation and license

Y. Feldman et al., *Massively parallel matrix-free 3D global linear stability analysis of
incompressible flows in OpenFOAM* (in preparation).

GPL-3.0-or-later, as OpenFOAM.
