#ifndef NewtonKrylov_H
#define NewtonKrylov_H

// Compact Krylov storage and batched global reductions for the Newton mAL
// solver.
//
// This deliberately does NOT live in src/matrixFreeFGMRESSafe: that library is
// shared with ArnoldiARPACK, msimplerNewton and msimplerShiftInvert, which are
// left bit-for-bit unchanged.

#include "MatrixFreeFlow.H"
#include "PtrList.H"
#include "fvMesh.H"
#include <mpi.h>

namespace Foam
{
namespace matrixFreeNewton
{

// ------------------------------------------------------------------------- //
// Batched global sum.
//
// Every Krylov dot product needs one global reduction.  Doing them one scalar
// at a time costs one collective per basis vector per Gram-Schmidt pass, which
// dominates the solver at large processor counts.  All dot products of one
// pass are accumulated locally and reduced with a single collective.
// ------------------------------------------------------------------------- //
inline void globalSumInPlace(scalar* values, const label n)
{
    static_assert
    (
        sizeof(scalar) == sizeof(double),
        "The batched Krylov reduction requires double precision"
    );

    if (n <= 0 || !Pstream::parRun()) return;

    // MPI_IN_PLACE expands to an old-style cast in the OpenMPI headers.
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wold-style-cast"
    MPI_Allreduce
    (
        MPI_IN_PLACE,
        values,
        int(n),
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );
    #pragma GCC diagnostic pop
}


// ------------------------------------------------------------------------- //
// Newton Krylov state.
//
// The state of the outer Newton FGMRES is (U, p) on the internal cells only:
//
//   * the metric reads internal U and p only,
//   * the mAL right preconditioner reads internal U and p only,
//   * the preconditioner output always satisfies z.phi == fvc::flux(z.U) and
//     homogeneous velocity boundary conditions, so both are reconstructed
//     exactly from U when the operator needs them,
//   * a basis vector's boundary values are never read.
//
// Storing the basis as Flow instead carried a surface flux (~3 values per cell
// on a hexahedral mesh) and three complete boundary fields per vector.
// ------------------------------------------------------------------------- //
struct StateVector
{
    vectorField U;
    scalarField p;

    explicit StateVector(const label nCells)
    :
        U(nCells, vector::zero),
        p(nCells, 0.0)
    {}
};

inline void zeroState(StateVector& a)
{
    a.U = vector::zero;
    a.p = scalar(0);
}

inline void copyState(StateVector& dst, const StateVector& src)
{
    dst.U = src.U;
    dst.p = src.p;
}

inline void scaleState(StateVector& a, const scalar s)
{
    forAll(a.U, celli)
    {
        a.U[celli] *= s;
        a.p[celli] *= s;
    }
}

inline void axpyState(StateVector& y, const scalar a, const StateVector& x)
{
    forAll(y.U, celli)
    {
        y.U[celli] += a*x.U[celli];
        y.p[celli] += a*x.p[celli];
    }
}

inline void loadState(StateVector& dst, const Flow& src)
{
    const vectorField& u = src.U.primitiveField();
    const scalarField& p = src.p.primitiveField();

    forAll(dst.U, celli)
    {
        dst.U[celli] = u[celli];
        dst.p[celli] = p[celli];
    }
}

// dst = state(A) - state(B)
inline void stateEqualsFlowMinusFlow
(
    StateVector& dst,
    const Flow& A,
    const Flow& B
)
{
    const vectorField& au = A.U.primitiveField();
    const scalarField& ap = A.p.primitiveField();
    const vectorField& bu = B.U.primitiveField();
    const scalarField& bp = B.p.primitiveField();

    forAll(dst.U, celli)
    {
        dst.U[celli] = au[celli] - bu[celli];
        dst.p[celli] = ap[celli] - bp[celli];
    }
}

// Write a Krylov state into the internal field of a full flow field.  The
// caller restores whatever boundary/flux consistency it needs.
inline void storeState(Flow& dst, const StateVector& src)
{
    vectorField& u = dst.U.primitiveFieldRef();
    scalarField& p = dst.p.primitiveFieldRef();

    forAll(src.U, celli)
    {
        u[celli] = src.U[celli];
        p[celli] = src.p[celli];
    }
}

inline void axpyStateIntoFlow(Flow& y, const scalar a, const StateVector& x)
{
    vectorField& u = y.U.primitiveFieldRef();
    scalarField& p = y.p.primitiveFieldRef();

    forAll(x.U, celli)
    {
        u[celli] += a*x.U[celli];
        p[celli] += a*x.p[celli];
    }
}


// ------------------------------------------------------------------------- //
// Workspaces.  Both Krylov bases are compact: the outer one stores (U,p), the
// inner one stores the single active velocity component.  The only full flow
// fields are the preconditioner input/output pair and the operator image.
// ------------------------------------------------------------------------- //
struct NewtonStateWorkspace
{
    label restart;
    PtrList<StateVector> V;
    PtrList<StateVector> Z;
    StateVector W;

    Flow precondInput;
    Flow precondOutput;
    Flow AX;

    List<scalar> H;
    List<scalar> cs;
    List<scalar> sn;
    List<scalar> g;
    List<scalar> y;
    List<scalar> dots;

    NewtonStateWorkspace
    (
        const word& prefix,
        const volVectorField& Utemplate,
        const surfaceScalarField& phitemplate,
        const volScalarField& ptemplate,
        const label restartSize
    )
    :
        restart(max(label(1), restartSize)),
        V(restart + 1),
        Z(restart),
        W(Utemplate.mesh().nCells()),
        precondInput(prefix + "_r", Utemplate, phitemplate, ptemplate),
        precondOutput(prefix + "_z", Utemplate, phitemplate, ptemplate),
        AX(prefix + "_AX", Utemplate, phitemplate, ptemplate),
        H((restart + 1)*restart, 0.0),
        cs(restart, 0.0),
        sn(restart, 0.0),
        g(restart + 1, 0.0),
        y(restart, 0.0),
        dots(restart + 1, 0.0)
    {
        const label nCells = Utemplate.mesh().nCells();

        for (label i=0; i<restart+1; ++i) V.set(i, new StateVector(nCells));
        for (label i=0; i<restart; ++i)   Z.set(i, new StateVector(nCells));

        zeroFlow(precondInput);
        zeroFlow(precondOutput);
        zeroFlow(AX);
    }
};

struct ComponentWorkspace
{
    label restart;
    PtrList<scalarField> V;
    PtrList<scalarField> Z;
    scalarField X;
    scalarField RHS;
    scalarField W;
    scalarField AX;

    List<scalar> H;
    List<scalar> cs;
    List<scalar> sn;
    List<scalar> g;
    List<scalar> y;
    List<scalar> dots;

    ComponentWorkspace(const label nCells, const label restartSize)
    :
        restart(max(label(1), restartSize)),
        V(restart + 1),
        Z(restart),
        X(nCells, 0.0),
        RHS(nCells, 0.0),
        W(nCells, 0.0),
        AX(nCells, 0.0),
        H((restart + 1)*restart, 0.0),
        cs(restart, 0.0),
        sn(restart, 0.0),
        g(restart + 1, 0.0),
        y(restart, 0.0),
        dots(restart + 1, 0.0)
    {
        for (label i=0; i<restart+1; ++i) V.set(i, new scalarField(nCells, 0.0));
        for (label i=0; i<restart; ++i)   Z.set(i, new scalarField(nCells, 0.0));
    }
};

} // namespace matrixFreeNewton
} // namespace Foam

#endif
