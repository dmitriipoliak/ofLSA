# matrixFreeBiCGStab2

OpenFOAM library for a matrix-free BiCGStab(2) solve on a coupled velocity-pressure correction vector

```text
x = (U, p)
```

The library does **not** assemble a matrix.  The user supplies a callback:

```cpp
void apply(const VpField& x, VpField& Ax) const;
```

For the SIMPLE-like Newton/Krylov action used in Gelfgat's lid-driven-cavity paper/code, the callback should:

1. Build the raw linearized momentum action from `x.U` and `x.p`.
2. Project the resulting velocity field with a Poisson equation.
3. Store the accumulated pressure correction in `Ax.p`.

Do **not** zero `Ax.p` at the end.  It is the projection pressure correction and becomes the pressure component of the Krylov vector.

## Build

Place this directory somewhere under your OpenFOAM source tree, for example:

```bash
~/OpenFOAM/owner-dev/src/matrixFreeBiCGStab2
```

Then build:

```bash
cd ~/OpenFOAM/owner-dev/src/matrixFreeBiCGStab2
./Allwmake
```

or:

```bash
wmake libso .
```

## Link from a solver

Add to your solver `Make/options`:

```make
EXE_INC = \
    -I$(WM_PROJECT_USER_DIR)/src/matrixFreeBiCGStab2 \
    -I$(LIB_SRC)/finiteVolume/lnInclude \
    -I$(LIB_SRC)/meshTools/lnInclude

EXE_LIBS = \
    -L$(FOAM_USER_LIBBIN) \
    -lmatrixFreeBiCGStab2 \
    -lfiniteVolume \
    -lmeshTools
```

Then include:

```cpp
#include "MatrixFreeBiCGStab2.H"
```

See `examples/usageInSolver.H` and `examples/ProjectedNewtonOperatorSkeleton.H`.

## Notes

The BiCGStab(2) implementation follows the BiCGStab(L) structure with `L=2`, requiring four matrix-vector products per outer iteration.  It uses a volume-weighted inner product by default:

```text
<U,p ; V,q> = sum_cells Vcell * (U.V + pressureWeight*p*q)
```

Set `pressureWeight` in the controls to rescale the pressure part if needed.
