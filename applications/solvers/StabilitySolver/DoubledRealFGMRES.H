#ifndef DoubledRealFGMRES_H
#define DoubledRealFGMRES_H

#include "ComplexFlow.H"
#include "PtrList.H"
#include "fvMesh.H"
#include "scalarField.H"

namespace Foam
{
namespace matrixFreeStability
{

// ------------------------------------------------------------------------- //
// Batched global sum.
//
// Every Krylov dot product needs one global reduction.  Doing them one scalar
// at a time costs one collective per dot product, which dominates the solver
// at large processor counts.  All orthogonalization dot products of one
// Gram-Schmidt pass are therefore accumulated locally and reduced with a
// single collective through this helper.
// ------------------------------------------------------------------------- //
void globalSumInPlace(scalar* values, const label n);


// ------------------------------------------------------------------------- //
// Doubled-real Krylov state.
//
// The Krylov state of the shifted saddle-point solve is (U_real, p_real,
// U_imag, p_imag) on the internal cells only.  Nothing else is needed:
//
//   * the metric (innerProduct/residualNorm) reads internal U and p only,
//   * the mAL right preconditioner reads internal U and p only,
//   * the preconditioner output always satisfies z.phi == fvc::flux(z.U) and
//     homogeneous velocity boundary conditions, so both are reconstructed
//     exactly from U when the operator needs them,
//   * a basis vector's boundary values are never read.
//
// Storing the basis as ComplexFlow instead carried a surface flux (~3 values
// per cell on a hexahedral mesh) and six complete boundary fields per vector.
// With restart+1 basis vectors plus restart preconditioned vectors that is
// ~112 bytes per cell per vector; the compact form is 64, i.e. ~40% less
// memory and ~40% less traffic through the O(restart^2) orthogonalization.
// ------------------------------------------------------------------------- //
struct ComplexStateVector
{
    vectorField Ur;
    vectorField Ui;
    scalarField pr;
    scalarField pi;

    explicit ComplexStateVector(const label nCells)
    :
        Ur(nCells, vector::zero),
        Ui(nCells, vector::zero),
        pr(nCells, 0.0),
        pi(nCells, 0.0)
    {}
};

inline void zeroComplexState(ComplexStateVector& a)
{
    a.Ur = vector::zero;
    a.Ui = vector::zero;
    a.pr = scalar(0);
    a.pi = scalar(0);
}

inline void copyComplexState
(
    ComplexStateVector& dst,
    const ComplexStateVector& src
)
{
    dst.Ur = src.Ur;
    dst.Ui = src.Ui;
    dst.pr = src.pr;
    dst.pi = src.pi;
}

inline void scaleComplexState(ComplexStateVector& a, const scalar s)
{
    forAll(a.Ur, celli)
    {
        a.Ur[celli] *= s;
        a.Ui[celli] *= s;
        a.pr[celli] *= s;
        a.pi[celli] *= s;
    }
}

inline void axpyComplexState
(
    ComplexStateVector& y,
    const scalar a,
    const ComplexStateVector& x
)
{
    forAll(y.Ur, celli)
    {
        y.Ur[celli] += a*x.Ur[celli];
        y.Ui[celli] += a*x.Ui[celli];
        y.pr[celli] += a*x.pr[celli];
        y.pi[celli] += a*x.pi[celli];
    }
}

// Extract the Krylov state of a full flow field.
inline void loadComplexState(ComplexStateVector& dst, const ComplexFlow& src)
{
    const vectorField& ur = src.real.U.primitiveField();
    const vectorField& ui = src.imag.U.primitiveField();
    const scalarField& pr = src.real.p.primitiveField();
    const scalarField& pi = src.imag.p.primitiveField();

    forAll(dst.Ur, celli)
    {
        dst.Ur[celli] = ur[celli];
        dst.Ui[celli] = ui[celli];
        dst.pr[celli] = pr[celli];
        dst.pi[celli] = pi[celli];
    }
}

// dst = state(A) - state(B)
inline void stateEqualsFlowMinusFlow
(
    ComplexStateVector& dst,
    const ComplexFlow& A,
    const ComplexFlow& B
)
{
    const vectorField& aur = A.real.U.primitiveField();
    const vectorField& aui = A.imag.U.primitiveField();
    const scalarField& apr = A.real.p.primitiveField();
    const scalarField& api = A.imag.p.primitiveField();

    const vectorField& bur = B.real.U.primitiveField();
    const vectorField& bui = B.imag.U.primitiveField();
    const scalarField& bpr = B.real.p.primitiveField();
    const scalarField& bpi = B.imag.p.primitiveField();

    forAll(dst.Ur, celli)
    {
        dst.Ur[celli] = aur[celli] - bur[celli];
        dst.Ui[celli] = aui[celli] - bui[celli];
        dst.pr[celli] = apr[celli] - bpr[celli];
        dst.pi[celli] = api[celli] - bpi[celli];
    }
}

// Write a Krylov state into the internal field of a full flow field.  The
// caller is responsible for whatever boundary/flux consistency it needs.
inline void storeComplexState(ComplexFlow& dst, const ComplexStateVector& src)
{
    vectorField& ur = dst.real.U.primitiveFieldRef();
    vectorField& ui = dst.imag.U.primitiveFieldRef();
    scalarField& pr = dst.real.p.primitiveFieldRef();
    scalarField& pi = dst.imag.p.primitiveFieldRef();

    forAll(src.Ur, celli)
    {
        ur[celli] = src.Ur[celli];
        ui[celli] = src.Ui[celli];
        pr[celli] = src.pr[celli];
        pi[celli] = src.pi[celli];
    }
}

inline void axpyStateIntoFlow
(
    ComplexFlow& y,
    const scalar a,
    const ComplexStateVector& x
)
{
    vectorField& ur = y.real.U.primitiveFieldRef();
    vectorField& ui = y.imag.U.primitiveFieldRef();
    scalarField& pr = y.real.p.primitiveFieldRef();
    scalarField& pi = y.imag.p.primitiveFieldRef();

    forAll(x.Ur, celli)
    {
        ur[celli] += a*x.Ur[celli];
        ui[celli] += a*x.Ui[celli];
        pr[celli] += a*x.pr[celli];
        pi[celli] += a*x.pi[celli];
    }
}


// Compact doubled-real vector used only by the inner mAL one-component
// velocity solves.  It stores exactly the active velocity component for the
// real and imaginary parts -- no pressure, no face flux and no inactive U
// components.
struct ComplexComponentVector
{
    scalarField real;
    scalarField imag;

    explicit ComplexComponentVector(const label nCells)
    :
        real(nCells, 0.0),
        imag(nCells, 0.0)
    {}
};

inline void zeroComplexComponent(ComplexComponentVector& a)
{
    a.real = scalar(0);
    a.imag = scalar(0);
}

inline void copyComplexComponent
(
    ComplexComponentVector& dst,
    const ComplexComponentVector& src
)
{
    dst.real = src.real;
    dst.imag = src.imag;
}

inline void scaleComplexComponent
(
    ComplexComponentVector& a,
    const scalar s
)
{
    a.real *= s;
    a.imag *= s;
}

inline void axpyComplexComponent
(
    ComplexComponentVector& y,
    const scalar a,
    const ComplexComponentVector& x
)
{
    forAll(y.real, celli)
    {
        y.real[celli] += a*x.real[celli];
        y.imag[celli] += a*x.imag[celli];
    }
}

inline void componentEqualsAminusB
(
    ComplexComponentVector& dst,
    const ComplexComponentVector& A,
    const ComplexComponentVector& B
)
{
    forAll(dst.real, celli)
    {
        dst.real[celli] = A.real[celli] - B.real[celli];
        dst.imag[celli] = A.imag[celli] - B.imag[celli];
    }
}

struct ComplexComponentFGMRESWorkspace
{
    label restart;
    PtrList<ComplexComponentVector> V;
    PtrList<ComplexComponentVector> Z;
    ComplexComponentVector X;
    ComplexComponentVector RHS;
    ComplexComponentVector W;
    ComplexComponentVector AX;
    List<scalar> H;
    List<scalar> cs;
    List<scalar> sn;
    List<scalar> g;
    List<scalar> y;
    // Scratch for the batched Gram-Schmidt reduction.
    List<scalar> dots;

    ComplexComponentFGMRESWorkspace
    (
        const label nCells,
        const label restartSize
    )
    :
        restart(max(label(1), restartSize)),
        V(restart + 1),
        Z(restart),
        X(nCells),
        RHS(nCells),
        W(nCells),
        AX(nCells),
        H((restart + 1)*restart, 0.0),
        cs(restart, 0.0),
        sn(restart, 0.0),
        g(restart + 1, 0.0),
        y(restart, 0.0),
        dots(restart + 1, 0.0)
    {
        for (label i=0; i<restart+1; ++i)
        {
            V.set(i, new ComplexComponentVector(nCells));
        }

        for (label i=0; i<restart; ++i)
        {
            Z.set(i, new ComplexComponentVector(nCells));
        }
    }
};

struct ComplexFGMRESWorkspace
{
    label restart;

    // Compact Krylov basis and preconditioned basis.
    PtrList<ComplexStateVector> V;
    PtrList<ComplexStateVector> Z;
    ComplexStateVector W;

    // The only full flow fields: the preconditioner input/output pair and the
    // operator image.  They are reused every iteration.
    ComplexFlow precondInput;
    ComplexFlow precondOutput;
    ComplexFlow AX;

    List<scalar> H;
    List<scalar> cs;
    List<scalar> sn;
    List<scalar> g;
    List<scalar> y;
    // Scratch for the batched Gram-Schmidt reduction.
    List<scalar> dots;

    ComplexFGMRESWorkspace
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

        for (label i=0; i<restart+1; ++i)
        {
            V.set(i, new ComplexStateVector(nCells));
        }

        for (label i=0; i<restart; ++i)
        {
            Z.set(i, new ComplexStateVector(nCells));
        }

        zeroComplexFlow(precondInput);
        zeroComplexFlow(precondOutput);
        zeroComplexFlow(AX);
    }
};


// ------------------------------------------------------------------------- //
// Shared state of the doubled-real Krylov solvers: the volume-weighted metric
// scaling, the stopping controls and the last-solve status.  Split out of
// DoubledRealFGMRESSolver so that the compact one-component solver can reuse
// it without inheriting (and having to stub out) the ComplexFlow operator
// interface it never uses.
// ------------------------------------------------------------------------- //
class DoubledRealKrylovBase
{
protected:
    const fvMesh& krylovMesh_;
    scalar volumeWeightScale_;

    label itMax_;
    scalar epsilon_;
    scalar absoluteTolerance_;
    // 1 = classical Gram-Schmidt, 2 = Gram-Schmidt with one reorthogonalization
    // (CGS2).  Two passes are required for a backward-stable basis and are the
    // configured default.
    label orthogonalizationPasses_;

    bool lastSolveConverged_;
    scalar lastInitialResidual_;
    scalar lastTrueResidual_;
    scalar lastTarget_;
    label lastIterations_;

public:
    DoubledRealKrylovBase
    (
        const fvMesh& mesh,
        const label itMax,
        const scalar epsilon,
        const scalar absoluteTolerance,
        const bool reorthogonalize
    );

    virtual ~DoubledRealKrylovBase() = default;

    bool lastSolveConverged() const { return lastSolveConverged_; }
    scalar lastInitialResidual() const { return lastInitialResidual_; }
    scalar lastTrueResidual() const { return lastTrueResidual_; }
    scalar lastTarget() const { return lastTarget_; }
    label lastIterations() const { return lastIterations_; }
};


class DoubledRealFGMRESSolver : public DoubledRealKrylovBase
{
protected:
    // Squared weight of the continuity (pressure) rows in the metric.
    scalar pressureWeightSquared_;

    virtual void rawMatrixAction
    (
        ComplexFlow& x,
        ComplexFlow& Ax
    ) = 0;

    virtual void applyRightPreconditioner
    (
        const ComplexFlow& r,
        ComplexFlow& z,
        const label iter
    ) = 0;

    // Complete doubled-real velocity/pressure metric in a single global
    // reduction.
    scalar innerProduct
    (
        const ComplexStateVector& A,
        const ComplexStateVector& B
    ) const;

    scalar residualNorm(const ComplexStateVector& A) const;

    // All n basis inner products <W,V[i]> in a single global reduction.
    void basisInnerProducts
    (
        const ComplexStateVector& W,
        const PtrList<ComplexStateVector>& V,
        const label n,
        scalar* dots
    ) const;

    // Restore the boundary/flux consistency that the operator needs after the
    // internal Krylov state of a full flow field has been updated.
    static void finaliseFlowState(ComplexFlow& x);

public:
    // Metric norm of a complete flow field (verification diagnostics only).
    scalar flowNorm(const ComplexFlow& A) const;

protected:

public:
    DoubledRealFGMRESSolver
    (
        const fvMesh& mesh,
        const label itMax,
        const scalar epsilon,
        const scalar absoluteTolerance,
        const bool reorthogonalize,
        const scalar pressureWeight = 1.0
    );

    void solveGMRES
    (
        ComplexFlow& x,
        const ComplexFlow& rhs,
        ComplexFGMRESWorkspace& ws,
        label& maxIterUsed,
        const bool useRelativeTolerance = true,
        const label printInterval = 10,
        const bool report = true,
        // x is known to be exactly zero on entry: the operator application to
        // it is skipped because it can only produce an exact zero.
        const bool zeroInitialGuess = false
    );
};

} // namespace matrixFreeStability
} // namespace Foam

#endif
