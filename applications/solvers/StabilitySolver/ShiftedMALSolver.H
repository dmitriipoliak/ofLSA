#ifndef ShiftedMALSolver_H
#define ShiftedMALSolver_H

#include "DoubledRealFGMRES.H"
#include "fvMatrices.H"
#include "autoPtr.H"

namespace Foam
{
namespace matrixFreeStability
{

class ShiftedMALSolver : public DoubledRealFGMRESSolver
{
    // Inner one-component doubled-real FGMRES.  It works entirely on compact
    // ComplexComponentVectors, so it needs the shared Krylov controls and
    // metric but none of the ComplexFlow operator interface.
    class MALComponentSolver : public DoubledRealKrylovBase
    {
        ShiftedMALSolver& owner_;
        direction cmpt_;

        scalar componentInnerProduct
        (
            const ComplexComponentVector& A,
            const ComplexComponentVector& B
        ) const;

        scalar componentResidualNorm
        (
            const ComplexComponentVector& A
        ) const;

        // All n basis inner products in a single global reduction.
        void componentBasisInnerProducts
        (
            const ComplexComponentVector& W,
            const PtrList<ComplexComponentVector>& V,
            const label n,
            scalar* dots
        ) const;

    public:
        void solveComponentGMRES
        (
            const ComplexComponentVector& rhs,
            ComplexComponentVector& x,
            ComplexComponentFGMRESWorkspace& ws,
            label& maxIterUsed,
            const label printInterval,
            const bool report
        );

        MALComponentSolver
        (
            ShiftedMALSolver& owner,
            const direction cmpt,
            const label maxIter,
            const scalar tolerance,
            const scalar absTolerance
        );
    };

    const Flow& base_;
    dimensionedScalar nu_;

    volScalarField& continuityWork_;
    surfaceVectorField& baseUFaceFrozen_;
    surfaceVectorField& momentumFluxWork_;

    volScalarField& mALPressurePoisson_;
    volScalarField& mALPressurePoissonStore_;

    label pRefCell_;
    scalar pRefValue_;

    scalar jacobianSign_;
    scalar sigmaReal_;
    scalar sigmaImag_;

    scalar malGamma_;
    label malVelocityBlockPrintInterval_;
    bool malReportVelocityBlock_;
    label malOseenShiftSweeps_;
    bool malRequireVelocityBlockConvergence_;

    // ---------------------------------------------------------------------- //
    // Run-constant geometry/base-state fields.
    //
    // Sf_i and the frozen base face velocity component U0_i(face) are needed by
    // every single-component operator application.  mesh.Sf().component(i) and
    // field.component(i) each allocate and fill a complete surface field, so
    // they are evaluated once and cached.
    // ---------------------------------------------------------------------- //
    PtrList<surfaceScalarField> sfComponent_;
    PtrList<surfaceScalarField> baseUFaceComponent_;

    // Per-component directional viscosity  nu + gamma (Sf_i/|Sf|)^2  and the
    // diagonal cross-convection flux  Sf_i U0_i(face).  Both depend only on the
    // mesh, nu, gamma and the frozen base face velocity, so they are constant
    // across the entire eigenvalue run.
    PtrList<surfaceScalarField> directionalNu_;
    PtrList<surfaceScalarField> directionalCrossPhi_;
    bool directionalFieldsValid_;

    // Per-component Oseen preconditioner matrices.  Their coefficients
    // (div(phi0) + div(crossPhi_i) - laplacian(nu_i), with the constant
    // -sigmaHatReal V diagonal shift) are constant across the whole run; only
    // the source changes between solves.  They are assembled once and reused.
    //
    // They are SCALAR matrices acting on the single active velocity component.
    // A vector matrix would make OpenFOAM run three segregated component
    // solves, two of them on an identically zero system, and still pay their
    // norm-factor reductions and matrix-vector products.
    PtrList<fvScalarMatrix> componentMatrix_;
    bool componentMatricesValid_;

    // Constant pressure-Laplacian matrix of the mAL Schur approximation.
    // Reassembling it on every preconditioner application (twice per FGMRES
    // iteration) is pure overhead: only its source changes.
    autoPtr<fvScalarMatrix> pressureMatrix_;
    scalar pressureReferenceSource_;

    ComplexFGMRESWorkspace workspace_;

    // One shared compact workspace for the sequential Ux/Uy/Uz inner solves.
    // Each Krylov vector stores only the active real/imag velocity component.
    ComplexComponentFGMRESWorkspace malComponentWorkspace_;

    autoPtr<MALComponentSolver> malXSolver_;
    autoPtr<MALComponentSolver> malYSolver_;
    autoPtr<MALComponentSolver> malZSolver_;

    ComplexFlow rhsBWork_;
    ComplexFlow malCanonicalRhsWork_;
    ComplexComponentVector malCompactRhsWork_;
    ComplexComponentVector malCompactSolutionWork_;
    ComplexFlow malVelocityActionWork_;
    ComplexFlow malPressureRhsWork_;

    // Scalar work fields of the single-component operator and of the Oseen
    // component solve.  The single-component action never touches the two
    // inactive velocity components, so it costs a third of the vector action
    // it replaces.
    volScalarField malOseenU_;
    volScalarField componentInputReal_;
    volScalarField componentInputImag_;
    surfaceScalarField componentPhiReal_;
    surfaceScalarField componentPhiImag_;
    surfaceScalarField componentFluxWork_;

    void rawMatrixAction
    (
        ComplexFlow& x,
        ComplexFlow& Ax
    ) override;

    void applyRightPreconditioner
    (
        const ComplexFlow& r,
        ComplexFlow& z,
        const label iter
    ) override;

    // Full Newton-Jacobian action, including the pressure gradient and the
    // continuity row.
    void applyLinearAction(const Flow& x, Flow& Ax) const;

    // Velocity-row-only action used inside mAL, where the input pressure is
    // identically zero and the output pressure/flux rows are discarded.
    void applyMALVelocityRowAction
    (
        const Flow& x,
        volVectorField& AxU
    ) const;

    void applyMALAugmentedVelocityAction
    (
        const ComplexFlow& x,
        ComplexFlow& Ax
    );

    // Single-component diagonal block action, evaluated with scalar fields.
    void applyMALDiagonalComponentAction
    (
        const direction cmpt,
        const ComplexComponentVector& x,
        ComplexComponentVector& Ax
    );

    void applyMALOseenComponentPreconditioner
    (
        const direction cmpt,
        const ComplexComponentVector& rhs,
        ComplexComponentVector& z
    );

    // (Re)build the cached components of the frozen base face velocity.
    void updateBaseFaceComponents();

    // (Re)build the cached directional-nu and cross-flux surface fields.
    void updateDirectionalFields();

    // (Re)assemble the cached per-component Oseen preconditioner matrices.
    void updateComponentMatrices();

    bool solveMALDiagonalComponent
    (
        const direction cmpt,
        const ComplexComponentVector& rhs,
        ComplexComponentVector& x,
        label& iterations
    );

    // One native GAMG cycle of the pressure Laplacian.  enforceReference
    // selects whether the pressure gauge is fixed on the result; intermediate
    // results whose gauge is removed by a later reference enforcement do not
    // need it.
    void solveMALPressurePoisson
    (
        const volScalarField& rhs,
        volScalarField& result,
        const bool enforceReference
    );

    void applyMALSchurInverse
    (
        const ComplexFlow& rhs,
        ComplexFlow& pressure
    );

    void applyMAL(const ComplexFlow& r, ComplexFlow& z);

public:
    ShiftedMALSolver
    (
        const Flow& base,
        const dimensionedScalar& nu,
        volScalarField& continuityWork,
        surfaceVectorField& baseUFaceFrozen,
        surfaceVectorField& momentumFluxWork,
        volScalarField& mALPressurePoisson,
        volScalarField& mALPressurePoissonStore,
        const label pRefCell,
        const scalar pRefValue,
        const scalar jacobianSign,
        const scalar sigmaReal,
        const scalar sigmaImag,
        const scalar continuityWeight,
        const label maxIter,
        const label restart,
        const scalar tolerance,
        const scalar absTolerance,
        const scalar malGamma,
        const label malVelocityBlockMaxIter,
        const label malVelocityBlockRestart,
        const scalar malVelocityBlockTolerance,
        const scalar malVelocityBlockAbsTolerance,
        const label malVelocityBlockPrintInterval,
        const label malOseenShiftSweeps,
        const bool malRequireVelocityBlockConvergence,
        const bool malReportVelocityBlock
    );

    ~ShiftedMALSolver();

    void updateFrozenBaseFaceVelocity();

    void setShift(const scalar sigmaReal, const scalar sigmaImag)
    {
        sigmaReal_ = sigmaReal;
        sigmaImag_ = sigmaImag;
        // The cached component matrices bake in the -sigmaHatReal V diagonal
        // shift, so they must be rebuilt when the shift changes.
        componentMatricesValid_ = false;
    }

    scalar sigmaReal() const { return sigmaReal_; }
    scalar sigmaImag() const { return sigmaImag_; }

    void applyB(const ComplexFlow& q, ComplexFlow& Bq) const;
    void applySignedJacobian(const ComplexFlow& q, ComplexFlow& Jq) const;
    void applyShiftedOperator(ComplexFlow& q, ComplexFlow& Aq);

    bool applyShiftInvert
    (
        const ComplexFlow& q,
        ComplexFlow& result,
        label& iterations,
        const label printInterval,
        const bool report = true
    );

    scalar norm(const ComplexFlow& q) const { return flowNorm(q); }
};

} // namespace matrixFreeStability
} // namespace Foam

#endif
