#include "ShiftedMALSolver.H"
#include "fvcGrad.H"
#include "fvcDiv.H"
#include "fvcFlux.H"
#include "fvcLaplacian.H"
#include "fvcSurfaceIntegrate.H"
#include "surfaceInterpolate.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "SolverPerformance.H"
#include "fixedValueFvPatchFields.H"
#include <cmath>

namespace Foam
{
namespace matrixFreeStability
{

using namespace matrixFreeNewton;

namespace
{
inline bool finiteScalarLocal(const scalar x)
{
    return std::isfinite(static_cast<double>(x));
}

void copyFlowComponentToCompact
(
    ComplexComponentVector& dst,
    const ComplexFlow& src,
    const direction cmpt
)
{
    const vectorField& ur = src.real.U.primitiveField();
    const vectorField& ui = src.imag.U.primitiveField();

    forAll(dst.real, celli)
    {
        dst.real[celli] = ur[celli][cmpt];
        dst.imag[celli] = ui[celli][cmpt];
    }
}

// dst = component cmpt of (A - B)
void flowComponentDifferenceToCompact
(
    ComplexComponentVector& dst,
    const ComplexFlow& A,
    const ComplexFlow& B,
    const direction cmpt
)
{
    const vectorField& ar = A.real.U.primitiveField();
    const vectorField& ai = A.imag.U.primitiveField();
    const vectorField& br = B.real.U.primitiveField();
    const vectorField& bi = B.imag.U.primitiveField();

    forAll(dst.real, celli)
    {
        dst.real[celli] = ar[celli][cmpt] - br[celli][cmpt];
        dst.imag[celli] = ai[celli][cmpt] - bi[celli][cmpt];
    }
}

void copyCompactToVelocityComponent
(
    ComplexFlow& dst,
    const ComplexComponentVector& src,
    const direction cmpt
)
{
    vectorField& ur = dst.real.U.primitiveFieldRef();
    vectorField& ui = dst.imag.U.primitiveFieldRef();

    forAll(src.real, celli)
    {
        ur[celli][cmpt] = src.real[celli];
        ui[celli][cmpt] = src.imag[celli];
    }
}

// Zero the internal velocity field only.  Every caller follows this with
// setHomogeneousVelocityBoundary(), which re-evaluates the coupled patches, so
// zeroing (and communicating) the boundary here would be wasted work.
void zeroInternalVelocity(ComplexFlow& a)
{
    vectorField& ur = a.real.U.primitiveFieldRef();
    vectorField& ui = a.imag.U.primitiveFieldRef();
    forAll(ur, celli)
    {
        ur[celli] = vector::zero;
        ui[celli] = vector::zero;
    }
}

// Zero a velocity field without evaluating boundary conditions.  The boundary
// values are still cleared so that repeated accumulation cannot drift, but the
// (communicating) coupled-patch evaluation is left to the single
// setHomogeneousVelocityBoundary() that every caller ends with.
// Copy/scale the velocity and pressure rows of a full flow field.  The mAL
// canonical right-hand side never reads the flux row of its argument, so the
// surface flux is left untouched.
void copyFlowStateRows(ComplexFlow& dst, const ComplexFlow& src)
{
    copyVolVector(dst.real.U, src.real.U);
    copyVolScalar(dst.real.p, src.real.p);
    copyVolVector(dst.imag.U, src.imag.U);
    copyVolScalar(dst.imag.p, src.imag.p);
}

void scaleFlowStateRows(ComplexFlow& a, const scalar s)
{
    for (label part=0; part<2; ++part)
    {
        Flow& f = part == 0 ? a.real : a.imag;

        vectorField& U = f.U.primitiveFieldRef();
        forAll(U, celli) U[celli] *= s;

        scalarField& p = f.p.primitiveFieldRef();
        forAll(p, celli) p[celli] *= s;

        forAll(f.U.boundaryFieldRef(), patchi)
        {
            auto& Ubf = f.U.boundaryFieldRef()[patchi];
            forAll(Ubf, i) Ubf[i] *= s;

            auto& pbf = f.p.boundaryFieldRef()[patchi];
            forAll(pbf, i) pbf[i] *= s;
        }
    }
}

void zeroVolVectorNoEvaluate(volVectorField& f)
{
    vectorField& fi = f.primitiveFieldRef();
    forAll(fi, celli) fi[celli] = vector::zero;

    forAll(f.boundaryFieldRef(), patchi)
    {
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = vector::zero;
    }
}

// Impose the homogeneous flux boundary condition without communication.
void zeroPhysicalSurfaceBoundary(surfaceScalarField& f)
{
    forAll(f.boundaryFieldRef(), patchi)
    {
        if (f.boundaryField()[patchi].coupled()) continue;
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = 0.0;
    }
}

void enforceZeroPressureReference(volScalarField& p, const label pRefCell)
{
    scalar localRef = 0.0;
    if (pRefCell >= 0) localRef = p.primitiveField()[pRefCell];

    globalSumInPlace(&localRef, 1);
    scalarField& pi = p.primitiveFieldRef();
    forAll(pi, celli) pi[celli] -= localRef;
    p.correctBoundaryConditions();
}

void copyInternalAndCorrect(volScalarField& dst, const volScalarField& src)
{
    scalarField& d = dst.primitiveFieldRef();
    const scalarField& s = src.primitiveField();
    forAll(d, celli) d[celli] = s[celli];
    dst.correctBoundaryConditions();
}

void copySurfVector(surfaceVectorField& dst, const surfaceVectorField& src)
{
    vectorField& dI = dst.primitiveFieldRef();
    const vectorField& sI = src.primitiveField();
    forAll(dI, facei) dI[facei] = sI[facei];

    forAll(dst.boundaryField(), patchi)
    {
        vectorField& dB = dst.boundaryFieldRef()[patchi];
        const vectorField& sB = src.boundaryField()[patchi];
        forAll(dB, facei) dB[facei] = sB[facei];
    }
}

void multiplySurfaceInto
(
    surfaceScalarField& dst,
    const surfaceScalarField& a,
    const surfaceScalarField& b
)
{
    scalarField& d = dst.primitiveFieldRef();
    const scalarField& ai = a.primitiveField();
    const scalarField& bi = b.primitiveField();
    forAll(d, facei) d[facei] = ai[facei]*bi[facei];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        scalarField& db = dst.boundaryFieldRef()[patchi];
        const scalarField& ab = a.boundaryField()[patchi];
        const scalarField& bb = b.boundaryField()[patchi];
        forAll(db, facei) db[facei] = ab[facei]*bb[facei];
    }
}

void addScaledProductToSurface
(
    surfaceScalarField& dst,
    const scalar coeff,
    const surfaceScalarField& a,
    const surfaceScalarField& b
)
{
    scalarField& d = dst.primitiveFieldRef();
    const scalarField& ai = a.primitiveField();
    const scalarField& bi = b.primitiveField();
    forAll(d, facei) d[facei] += coeff*ai[facei]*bi[facei];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        scalarField& db = dst.boundaryFieldRef()[patchi];
        const scalarField& ab = a.boundaryField()[patchi];
        const scalarField& bb = b.boundaryField()[patchi];
        forAll(db, facei) db[facei] += coeff*ab[facei]*bb[facei];
    }
}

tmp<volVectorField> frozenBaseVelocityFluxDivergence
(
    const surfaceScalarField& perturbationPhi,
    const surfaceVectorField& baseUFaceFrozen,
    surfaceVectorField& momentumFluxWork
)
{
    vectorField& mI = momentumFluxWork.primitiveFieldRef();
    const scalarField& phiI = perturbationPhi.primitiveField();
    const vectorField& uFI = baseUFaceFrozen.primitiveField();
    forAll(mI, facei) mI[facei] = phiI[facei]*uFI[facei];

    forAll(momentumFluxWork.boundaryField(), patchi)
    {
        vectorField& mB = momentumFluxWork.boundaryFieldRef()[patchi];
        const scalarField& phiB = perturbationPhi.boundaryField()[patchi];
        const vectorField& uFB = baseUFaceFrozen.boundaryField()[patchi];
        forAll(mB, facei) mB[facei] = phiB[facei]*uFB[facei];
    }

    return fvc::div(momentumFluxWork);
}

// Patch types of a scalar field that represents one Cartesian component of the
// homogeneous velocity perturbation.  Coupled patches keep the velocity patch
// type; physical patches must be representable for a scalar field, otherwise
// the single-component operator could not reproduce the velocity operator.
wordList velocityComponentPatchTypes(const volVectorField& U)
{
    const fvMesh& mesh = U.mesh();
    wordList patchTypes(mesh.boundary().size());

    forAll(mesh.boundary(), patchi)
    {
        const word& type = U.boundaryField()[patchi].type();

        if (!fvPatchField<scalar>::patchConstructorTablePtr_->found(type))
        {
            FatalErrorInFunction
                << "Velocity patch type " << type << " on patch "
                << mesh.boundary()[patchi].name()
                << " has no scalar counterpart, so the mAL single-component "
                << "velocity operator cannot represent it." << nl
                << "Use component-wise velocity boundary conditions "
                << "(e.g. fixedValue/zeroGradient)."
                << exit(FatalError);
        }

        patchTypes[patchi] = type;
    }

    return patchTypes;
}

// Zero every boundary value once.  Physical patch values of the perturbation
// component are identically zero for the whole run, so this never needs
// repeating; coupled patches are refreshed by correctBoundaryConditions().
void initialiseComponentField(volScalarField& f)
{
    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, celli) fi[celli] = 0.0;

    forAll(f.boundaryFieldRef(), patchi)
    {
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = 0.0;
    }

    f.correctBoundaryConditions();
}

// Load one Cartesian velocity component into its scalar carrier.  Only the
// coupled patches need refreshing: the physical patch values of the
// homogeneous perturbation are identically zero for the whole run.
void loadComponentField(volScalarField& f, const scalarField& src)
{
    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, celli) fi[celli] = src[celli];
    f.correctBoundaryConditions();
}
}

// ------------------------------------------------------------------------- //
// Inner component solver.  Its exact action contains the full complex shift.
// The deepest velocity preconditioner is also shift-aware, but remains purely
// OpenFOAM: a few doubled-real block Gauss-Seidel sweeps whose diagonal solves
// use the existing mALOseen_U PBiCGStab/DILU solver.  Native OpenFOAM GAMG is
// used only by the pressure L^{-1} action in the mAL Schur approximation.
// ------------------------------------------------------------------------- //

ShiftedMALSolver::MALComponentSolver::MALComponentSolver
(
    ShiftedMALSolver& owner,
    const direction cmpt,
    const label maxIter,
    const scalar tolerance,
    const scalar absTolerance
)
:
    DoubledRealKrylovBase
    (
        owner.base_.U.mesh(),
        maxIter,
        tolerance,
        absTolerance,
        true
    ),
    owner_(owner),
    cmpt_(cmpt)
{}


scalar ShiftedMALSolver::MALComponentSolver::componentInnerProduct
(
    const ComplexComponentVector& A,
    const ComplexComponentVector& B
) const
{
    scalar local = 0.0;
    const scalarField& V = krylovMesh_.V();

    forAll(A.real, celli)
    {
        local += volumeWeightScale_*V[celli]*
        (
            A.real[celli]*B.real[celli]
          + A.imag[celli]*B.imag[celli]
        );
    }

    globalSumInPlace(&local, 1);
    return local;
}


scalar ShiftedMALSolver::MALComponentSolver::componentResidualNorm
(
    const ComplexComponentVector& A
) const
{
    const scalar v = componentInnerProduct(A, A);
    if (!finiteScalarLocal(v) || v < 0.0) return GREAT;
    return Foam::sqrt(max(v, scalar(0)));
}


void ShiftedMALSolver::MALComponentSolver::componentBasisInnerProducts
(
    const ComplexComponentVector& W,
    const PtrList<ComplexComponentVector>& V,
    const label n,
    scalar* dots
) const
{
    const scalarField& cellV = krylovMesh_.V();

    for (label i=0; i<n; ++i)
    {
        const ComplexComponentVector& Vi = V[i];

        scalar local = 0.0;
        forAll(W.real, celli)
        {
            local += volumeWeightScale_*cellV[celli]*
            (
                W.real[celli]*Vi.real[celli]
              + W.imag[celli]*Vi.imag[celli]
            );
        }

        dots[i] = local;
    }

    globalSumInPlace(dots, n);
}


void ShiftedMALSolver::MALComponentSolver::solveComponentGMRES
(
    const ComplexComponentVector& rhs,
    ComplexComponentVector& x,
    ComplexComponentFGMRESWorkspace& ws,
    label& maxIterUsed,
    const label printInterval,
    const bool report
)
{
    const label restart = ws.restart;
    const label printEvery = max(label(1), printInterval);

    auto H = [&](const label row, const label col) -> scalar&
    {
        return ws.H[row + (restart + 1)*col];
    };

    // The solve always starts from x=0 and the action is linear, so A*x is
    // exactly zero and r0 is exactly rhs.
    zeroComplexComponent(ws.X);
    copyComplexComponent(ws.RHS, rhs);
    copyComplexComponent(ws.W, rhs);

    const scalar initialResidual = componentResidualNorm(ws.W);
    const scalar target = max(epsilon_*max(initialResidual, SMALL), absoluteTolerance_);

    const scalar residualScale =
        (finiteScalarLocal(initialResidual) && initialResidual > VSMALL)
      ? initialResidual
      : scalar(1);
    const scalar normalizedTarget = target/residualScale;

    lastSolveConverged_ = false;
    lastInitialResidual_ = initialResidual/residualScale;
    lastTrueResidual_ = lastInitialResidual_;
    lastTarget_ = normalizedTarget;
    lastIterations_ = 0;

    if (report)
    {
        Info<< "FGMRES(mAL-component): initialResidual="
            << lastInitialResidual_
            << " target=" << normalizedTarget
            << " restart=" << restart
            << " maxIter=" << itMax_ << nl;
    }

    if (initialResidual <= target)
    {
        maxIterUsed = 0;
        lastSolveConverged_ = true;
        copyComplexComponent(x, ws.X);
        return;
    }

    label globalIter = 0;
    scalar trueResidual = initialResidual;

    while (globalIter < itMax_ && trueResidual > target)
    {
        forAll(ws.H, i) ws.H[i] = 0.0;
        forAll(ws.cs, i) ws.cs[i] = 0.0;
        forAll(ws.sn, i) ws.sn[i] = 0.0;
        forAll(ws.g, i) ws.g[i] = 0.0;
        forAll(ws.y, i) ws.y[i] = 0.0;

        const scalar beta = componentResidualNorm(ws.W);
        if (!finiteScalarLocal(beta) || beta < VSMALL) break;

        copyComplexComponent(ws.V[0], ws.W);
        scaleComplexComponent(ws.V[0], 1.0/beta);
        ws.g[0] = beta;

        label m = 0;

        for (label j=0; j<restart && globalIter<itMax_; ++j)
        {
            ++globalIter;

            owner_.applyMALOseenComponentPreconditioner
            (
                cmpt_,
                ws.V[j],
                ws.Z[j]
            );
            owner_.applyMALDiagonalComponentAction(cmpt_, ws.Z[j], ws.W);

            // Gram-Schmidt with one reorthogonalization (CGS2): one global
            // reduction per pass instead of one per basis vector.
            for (label pass=0; pass<orthogonalizationPasses_; ++pass)
            {
                componentBasisInnerProducts
                (
                    ws.W,
                    ws.V,
                    j+1,
                    ws.dots.begin()
                );

                for (label i=0; i<=j; ++i)
                {
                    scalar h = ws.dots[i];
                    if (!finiteScalarLocal(h)) h = 0.0;

                    H(i,j) += h;
                    axpyComplexComponent(ws.W, -h, ws.V[i]);
                }
            }

            H(j+1,j) = componentResidualNorm(ws.W);
            if (finiteScalarLocal(H(j+1,j)) && H(j+1,j) > VSMALL)
            {
                copyComplexComponent(ws.V[j+1], ws.W);
                scaleComplexComponent(ws.V[j+1], 1.0/H(j+1,j));
            }
            else
            {
                H(j+1,j) = 0.0;
                zeroComplexComponent(ws.V[j+1]);
            }

            for (label i=0; i<j; ++i)
            {
                const scalar hij = H(i,j);
                const scalar hi1j = H(i+1,j);
                H(i,j) = ws.cs[i]*hij + ws.sn[i]*hi1j;
                H(i+1,j) = -ws.sn[i]*hij + ws.cs[i]*hi1j;
            }

            const scalar h1 = H(j,j);
            const scalar h2 = H(j+1,j);
            const scalar den = scalar
            (
                std::hypot
                (
                    static_cast<double>(h1),
                    static_cast<double>(h2)
                )
            );

            if (finiteScalarLocal(den) && den > VSMALL)
            {
                ws.cs[j] = h1/den;
                ws.sn[j] = h2/den;
            }
            else
            {
                ws.cs[j] = 1.0;
                ws.sn[j] = 0.0;
            }

            H(j,j) = ws.cs[j]*h1 + ws.sn[j]*h2;
            H(j+1,j) = 0.0;

            const scalar gj = ws.g[j];
            ws.g[j] = ws.cs[j]*gj;
            ws.g[j+1] = -ws.sn[j]*gj;

            m = j+1;
            const scalar estimatedResidual = mag(ws.g[j+1]);

            if
            (
                report
             &&
                (
                    globalIter % printEvery == 0
                 || estimatedResidual <= target
                )
            )
            {
                Info<< "FGMRES(mAL-component): iter=" << globalIter
                    << " estimatedResidual="
                    << estimatedResidual/residualScale
                    << " target=" << normalizedTarget << nl;
            }

            if
            (
                !finiteScalarLocal(estimatedResidual)
             || estimatedResidual <= target
            )
            {
                break;
            }
        }

        for (label i=m-1; i>=0; --i)
        {
            scalar sum = ws.g[i];
            for (label k=i+1; k<m; ++k) sum -= H(i,k)*ws.y[k];

            if
            (
                !finiteScalarLocal(sum)
             || !finiteScalarLocal(H(i,i))
             || mag(H(i,i)) < VSMALL
            )
            {
                ws.y[i] = 0.0;
            }
            else
            {
                ws.y[i] = sum/H(i,i);
                if (!finiteScalarLocal(ws.y[i])) ws.y[i] = 0.0;
            }
        }

        for (label i=0; i<m; ++i)
        {
            if (finiteScalarLocal(ws.y[i]))
            {
                axpyComplexComponent(ws.X, ws.y[i], ws.Z[i]);
            }
        }

        owner_.applyMALDiagonalComponentAction(cmpt_, ws.X, ws.AX);
        componentEqualsAminusB(ws.W, ws.RHS, ws.AX);
        trueResidual = componentResidualNorm(ws.W);

        if (report)
        {
            Info<< "FGMRES(mAL-component): restart end iter=" << globalIter
                << " trueResidual=" << trueResidual/residualScale
                << " target=" << normalizedTarget << nl;
        }
    }

    maxIterUsed = globalIter;
    lastIterations_ = globalIter;
    lastTrueResidual_ = trueResidual/residualScale;
    lastSolveConverged_ =
        finiteScalarLocal(trueResidual) && trueResidual <= target;

    copyComplexComponent(x, ws.X);

    if (report)
    {
        Info<< "FGMRES(mAL-component): "
            << (lastSolveConverged_ ? "CONVERGED" : "FAILED")
            << " iterations=" << globalIter
            << " trueResidual=" << trueResidual/residualScale
            << " target=" << normalizedTarget << nl;
    }
}

// ------------------------------------------------------------------------- //

ShiftedMALSolver::ShiftedMALSolver
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
)
:
    DoubledRealFGMRESSolver
    (
        base.U.mesh(),
        maxIter,
        tolerance,
        absTolerance,
        true,
        continuityWeight
    ),
    base_(base),
    nu_(nu),
    continuityWork_(continuityWork),
    baseUFaceFrozen_(baseUFaceFrozen),
    momentumFluxWork_(momentumFluxWork),
    mALPressurePoisson_(mALPressurePoisson),
    mALPressurePoissonStore_(mALPressurePoissonStore),
    pRefCell_(pRefCell),
    pRefValue_(pRefValue),
    jacobianSign_(jacobianSign),
    sigmaReal_(sigmaReal),
    sigmaImag_(sigmaImag),
    malGamma_(malGamma),
    malVelocityBlockPrintInterval_(max(label(1), malVelocityBlockPrintInterval)),
    malReportVelocityBlock_(malReportVelocityBlock),
    malOseenShiftSweeps_(max(label(1), malOseenShiftSweeps)),
    malRequireVelocityBlockConvergence_(malRequireVelocityBlockConvergence),
    sfComponent_(3),
    baseUFaceComponent_(3),
    directionalNu_(),
    directionalCrossPhi_(),
    directionalFieldsValid_(false),
    componentMatrix_(),
    componentMatricesValid_(false),
    pressureMatrix_(),
    pressureReferenceSource_(0.0),
    workspace_("ShiftedGMRES", base.U, base.phi, base.p, restart),
    malComponentWorkspace_
    (
        base.U.mesh().nCells(),
        min(malVelocityBlockRestart, malVelocityBlockMaxIter)
    ),
    malXSolver_(),
    malYSolver_(),
    malZSolver_(),
    rhsBWork_("Bq", base.U, base.phi, base.p),
    malCanonicalRhsWork_("mAL_canonicalRhs", base.U, base.phi, base.p),
    malCompactRhsWork_(base.U.mesh().nCells()),
    malCompactSolutionWork_(base.U.mesh().nCells()),
    malVelocityActionWork_("mAL_velocityAction", base.U, base.phi, base.p),
    malPressureRhsWork_("mAL_pressureRhs", base.U, base.phi, base.p),
    malOseenU_
    (
        IOobject
        (
            // The fvSolution entry is looked up by field name.
            "mALOseen_U",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.U.mesh(),
        dimensionedScalar("mALOseen_U", base.U.dimensions(), 0.0),
        velocityComponentPatchTypes(base.U)
    ),
    componentInputReal_
    (
        IOobject
        (
            "mALComponentInputReal",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.U.mesh(),
        dimensionedScalar("mALComponentInputReal", base.U.dimensions(), 0.0),
        velocityComponentPatchTypes(base.U)
    ),
    componentInputImag_
    (
        IOobject
        (
            "mALComponentInputImag",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.U.mesh(),
        dimensionedScalar("mALComponentInputImag", base.U.dimensions(), 0.0),
        velocityComponentPatchTypes(base.U)
    ),
    componentPhiReal_
    (
        IOobject
        (
            "mALComponentPhiReal",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.phi
    ),
    componentPhiImag_
    (
        IOobject
        (
            "mALComponentPhiImag",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.phi
    ),
    componentFluxWork_
    (
        IOobject
        (
            "mALComponentFluxWork",
            base.U.mesh().time().name(),
            base.U.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        base.phi
    )
{
    if (jacobianSign_ != 1.0 && jacobianSign_ != -1.0)
    {
        FatalErrorInFunction
            << "jacobianSign must be +1 or -1" << exit(FatalError);
    }
    if
    (
        continuityWeight <= 0.0
     || malGamma_ < 0.0
     || malOseenShiftSweeps_ < 1
    )
    {
        FatalErrorInFunction << "Invalid stability/mAL controls"
            << exit(FatalError);
    }

    malXSolver_.reset
    (
        new MALComponentSolver
        (
            *this,
            0,
            malVelocityBlockMaxIter,
            malVelocityBlockTolerance,
            malVelocityBlockAbsTolerance
        )
    );
    malYSolver_.reset
    (
        new MALComponentSolver
        (
            *this,
            1,
            malVelocityBlockMaxIter,
            malVelocityBlockTolerance,
            malVelocityBlockAbsTolerance
        )
    );
    malZSolver_.reset
    (
        new MALComponentSolver
        (
            *this,
            2,
            malVelocityBlockMaxIter,
            malVelocityBlockTolerance,
            malVelocityBlockAbsTolerance
        )
    );

    zeroComplexFlow(rhsBWork_);
    zeroComplexFlow(malCanonicalRhsWork_);
    zeroComplexComponent(malCompactRhsWork_);
    zeroComplexComponent(malCompactSolutionWork_);
    zeroComplexFlow(malVelocityActionWork_);
    zeroComplexFlow(malPressureRhsWork_);

    initialiseComponentField(malOseenU_);
    initialiseComponentField(componentInputReal_);
    initialiseComponentField(componentInputImag_);
    zeroSurfScalar(componentPhiReal_);
    zeroSurfScalar(componentPhiImag_);
    zeroSurfScalar(componentFluxWork_);

    // Sf_i is pure geometry and never changes.
    const fvMesh& mesh = base_.U.mesh();
    for (direction cmpt=0; cmpt<3; ++cmpt)
    {
        sfComponent_.set
        (
            cmpt,
            new surfaceScalarField
            (
                IOobject
                (
                    "mALSf" + name(label(cmpt)),
                    mesh.time().name(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh.Sf().component(cmpt)
            )
        );
    }
}


ShiftedMALSolver::~ShiftedMALSolver() = default;


void ShiftedMALSolver::updateFrozenBaseFaceVelocity()
{
    const fvMesh& mesh = base_.U.mesh();

    tmp<surfaceVectorField> tBaseUFace = fvc::interpolate(base_.U);
    copySurfVector(baseUFaceFrozen_, tBaseUFace());
    tBaseUFace.clear();

    updateBaseFaceComponents();

    // The frozen base face velocity changed, so the cached directional cross
    // flux is stale.  Rebuild both cached directional fields now.
    updateDirectionalFields();
}


void ShiftedMALSolver::updateBaseFaceComponents()
{
    const fvMesh& mesh = base_.U.mesh();

    for (direction cmpt=0; cmpt<3; ++cmpt)
    {
        if (!baseUFaceComponent_.set(cmpt))
        {
            baseUFaceComponent_.set
            (
                cmpt,
                new surfaceScalarField
                (
                    IOobject
                    (
                        "mALBaseUFace" + name(label(cmpt)),
                        mesh.time().name(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    baseUFaceFrozen_.component(cmpt)
                )
            );
        }
        else
        {
            baseUFaceComponent_[cmpt] = baseUFaceFrozen_.component(cmpt);
        }
    }
}


void ShiftedMALSolver::updateDirectionalFields()
{
    const fvMesh& mesh = base_.U.mesh();

    // Defensive: the driver always calls updateFrozenBaseFaceVelocity() first,
    // but the directional cross flux needs the base face components.
    if (!baseUFaceComponent_.set(0)) updateBaseFaceComponents();

    const dimensionedScalar gammaNu
    (
        "mALDirectionalGamma",
        nu_.dimensions(),
        malGamma_
    );

    if (directionalNu_.empty())
    {
        directionalNu_.setSize(3);
        directionalCrossPhi_.setSize(3);
    }

    for (direction cmpt=0; cmpt<3; ++cmpt)
    {
        tmp<surfaceScalarField> tNu
        (
            nu_ + gammaNu*sqr(sfComponent_[cmpt]/mesh.magSf())
        );
        tmp<surfaceScalarField> tCross
        (
            sfComponent_[cmpt]*baseUFaceComponent_[cmpt]
        );

        if (!directionalNu_.set(cmpt))
        {
            directionalNu_.set
            (
                cmpt,
                new surfaceScalarField
                (
                    IOobject
                    (
                        "mALDirectionalNu" + name(label(cmpt)),
                        mesh.time().name(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    tNu
                )
            );
            directionalCrossPhi_.set
            (
                cmpt,
                new surfaceScalarField
                (
                    IOobject
                    (
                        "mALDirectionalCrossPhi" + name(label(cmpt)),
                        mesh.time().name(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    tCross
                )
            );
        }
        else
        {
            directionalNu_[cmpt] = tNu();
            directionalCrossPhi_[cmpt] = tCross();
        }
    }

    directionalFieldsValid_ = true;

    // The cached component matrices depend on these fields, so they are now
    // stale and must be reassembled before the next preconditioner solve.
    componentMatricesValid_ = false;
}


void ShiftedMALSolver::updateComponentMatrices()
{
    if (!directionalFieldsValid_) updateDirectionalFields();

    const fvMesh& mesh = base_.U.mesh();
    const scalar sigmaHatReal = jacobianSign_*sigmaReal_;
    const scalarField& V = mesh.V();

    // The boundary coefficients are taken from the (identically zero)
    // homogeneous perturbation boundary values, so assemble from a zero field.
    initialiseComponentField(malOseenU_);

    // Rebuild from scratch: cheap (done at most a handful of times per run) and
    // avoids any fvMatrix operator= constraints on the stored psi reference.
    componentMatrix_.clear();
    componentMatrix_.setSize(3);

    for (direction cmpt=0; cmpt<3; ++cmpt)
    {
        // Assemble the constant operator with NO source term.  Because the
        // implicit div/laplacian operators contribute nothing to source_, the
        // freshly built source_ is exactly zero; the solve then writes
        // V*rhs into it, reproducing "== rhs".
        tmp<fvScalarMatrix> tUEqn
        (
            fvm::div(base_.phi, malOseenU_)
          + fvm::div(directionalCrossPhi_[cmpt], malOseenU_)
          - fvm::laplacian(directionalNu_[cmpt], malOseenU_)
        );

        // Bake in the constant strong-form shift -sigmaHatReal*u, i.e. the FV
        // diagonal contribution -sigmaHatReal*V_i.
        scalarField& diag = tUEqn.ref().diag();
        forAll(diag, celli)
        {
            diag[celli] -= sigmaHatReal*V[celli];
        }

        componentMatrix_.set(cmpt, tUEqn.ptr());
    }

    componentMatricesValid_ = true;
}


void ShiftedMALSolver::applyLinearAction(const Flow& x, Flow& Ax) const
{
    // Ax.phi and Ax.p are overwritten in full below, so only the velocity row
    // needs clearing, and its coupled boundary values are re-evaluated by the
    // single setHomogeneousVelocityBoundary() of applyShiftedOperator().
    zeroVolVectorNoEvaluate(Ax.U);

    tmp<volVectorField> tGradP = fvc::grad(x.p);
    tmp<volVectorField> tLapU = fvc::laplacian(nu_, x.U);
    tmp<volVectorField> tConvBase = fvc::div(base_.phi, x.U);
    tmp<volVectorField> tConvNewton =
        frozenBaseVelocityFluxDivergence
        (
            x.phi,
            baseUFaceFrozen_,
            momentumFluxWork_
        );

    addFromTmp(Ax.U, tGradP, 1.0);
    addFromTmp(Ax.U, tLapU, -1.0);
    addFromTmp(Ax.U, tConvBase, 1.0);
    addFromTmp(Ax.U, tConvNewton, 1.0);

    copySurfScalar(Ax.phi, x.phi);
    zeroPhysicalSurfaceBoundary(Ax.phi);

    tmp<volScalarField> tDiv = fvc::div(Ax.phi);
    copyInternalAndCorrect(Ax.p, tDiv());
}


void ShiftedMALSolver::applyMALVelocityRowAction
(
    const Flow& x,
    volVectorField& AxU
) const
{
    // Inside mAL the input pressure is identically zero and the output
    // pressure/flux rows are discarded, so grad(p), the flux copy and
    // div(phi) of the full action are all skipped.
    zeroVolVectorNoEvaluate(AxU);

    tmp<volVectorField> tLapU = fvc::laplacian(nu_, x.U);
    tmp<volVectorField> tConvBase = fvc::div(base_.phi, x.U);
    tmp<volVectorField> tConvNewton =
        frozenBaseVelocityFluxDivergence
        (
            x.phi,
            baseUFaceFrozen_,
            momentumFluxWork_
        );

    addFromTmp(AxU, tLapU, -1.0);
    addFromTmp(AxU, tConvBase, 1.0);
    addFromTmp(AxU, tConvNewton, 1.0);
}


void ShiftedMALSolver::applySignedJacobian
(
    const ComplexFlow& q,
    ComplexFlow& Jq
) const
{
    applyLinearAction(q.real, Jq.real);
    applyLinearAction(q.imag, Jq.imag);

    if (jacobianSign_ < 0.0)
    {
        scaleComplexFlow(Jq, -1.0);
    }
}


void ShiftedMALSolver::applyB(const ComplexFlow& q, ComplexFlow& Bq) const
{
    zeroComplexFlow(Bq);
    copyVolVector(Bq.real.U, q.real.U);
    copyVolVector(Bq.imag.U, q.imag.U);
    setHomogeneousVelocityBoundary(Bq);
}


void ShiftedMALSolver::applyShiftedOperator
(
    ComplexFlow& q,
    ComplexFlow& Aq
)
{
    applySignedJacobian(q, Aq);

    // (J - sigma B)(q_r+i q_i), B=I only in velocity rows.
    axpyVolVector(Aq.real.U, -sigmaReal_, q.real.U);
    axpyVolVector(Aq.real.U,  sigmaImag_, q.imag.U);
    axpyVolVector(Aq.imag.U, -sigmaReal_, q.imag.U);
    axpyVolVector(Aq.imag.U, -sigmaImag_, q.real.U);

    setHomogeneousVelocityBoundary(Aq);
}


void ShiftedMALSolver::rawMatrixAction
(
    ComplexFlow& x,
    ComplexFlow& Ax
)
{
    applyShiftedOperator(x, Ax);
}


void ShiftedMALSolver::applyRightPreconditioner
(
    const ComplexFlow& r,
    ComplexFlow& z,
    const label iter
)
{
    (void)iter;
    applyMAL(r, z);
}


void ShiftedMALSolver::applyMALAugmentedVelocityAction
(
    const ComplexFlow& x,
    ComplexFlow& Ax
)
{
    // x already carries homogeneous velocity boundaries and phi=fvc::flux(U).
    // Only the velocity rows of Ax are written; its pressure/flux rows are
    // never read by the callers.
    applyMALVelocityRowAction(x.real, Ax.real.U);
    applyMALVelocityRowAction(x.imag, Ax.imag.U);

    if (malGamma_ > SMALL)
    {
        tmp<volScalarField> tDivReal = fvc::div(x.real.phi);
        tmp<volScalarField> tDivImag = fvc::div(x.imag.phi);
        tmp<volVectorField> tGradDivReal = fvc::grad(tDivReal());
        tmp<volVectorField> tGradDivImag = fvc::grad(tDivImag());

        addFromTmp(Ax.real.U, tGradDivReal, -malGamma_);
        addFromTmp(Ax.imag.U, tGradDivImag, -malGamma_);
    }

    // Canonical system: J_N - sigmaHat B, sigmaHat=jacobianSign*sigma.
    const scalar sigmaHatReal = jacobianSign_*sigmaReal_;
    const scalar sigmaHatImag = jacobianSign_*sigmaImag_;

    axpyVolVector(Ax.real.U, -sigmaHatReal, x.real.U);
    axpyVolVector(Ax.real.U,  sigmaHatImag, x.imag.U);
    axpyVolVector(Ax.imag.U, -sigmaHatReal, x.imag.U);
    axpyVolVector(Ax.imag.U, -sigmaHatImag, x.real.U);
}


void ShiftedMALSolver::applyMALDiagonalComponentAction
(
    const direction cmpt,
    const ComplexComponentVector& x,
    ComplexComponentVector& Ax
)
{
    // Diagonal block of the augmented velocity operator for one Cartesian
    // component.  The two inactive components of the input are identically
    // zero and the two inactive output components are discarded, so the whole
    // action is evaluated on scalar fields.  Every operator involved (Gauss
    // div/laplacian/grad, linear interpolation) acts component-wise with the
    // same face weights, so this reproduces the corresponding component of the
    // full vector action; the two surface integrals below are summed before
    // integration, which is equivalent up to rounding.
    loadComponentField(componentInputReal_, x.real);
    loadComponentField(componentInputImag_, x.imag);

    const surfaceScalarField& Sfi = sfComponent_[cmpt];
    const surfaceScalarField& U0i = baseUFaceComponent_[cmpt];

    // phi' = fvc::flux(u e_i) = interpolate(u) Sf_i
    {
        tmp<surfaceScalarField> tIr = fvc::interpolate(componentInputReal_);
        multiplySurfaceInto(componentPhiReal_, tIr(), Sfi);
        tIr.clear();

        tmp<surfaceScalarField> tIi = fvc::interpolate(componentInputImag_);
        multiplySurfaceInto(componentPhiImag_, tIi(), Sfi);
    }

    const scalar sigmaHatReal = jacobianSign_*sigmaReal_;
    const scalar sigmaHatImag = jacobianSign_*sigmaImag_;

    for (label part=0; part<2; ++part)
    {
        const volScalarField& u =
            part == 0 ? componentInputReal_ : componentInputImag_;
        const volScalarField& uOther =
            part == 0 ? componentInputImag_ : componentInputReal_;
        const surfaceScalarField& phiP =
            part == 0 ? componentPhiReal_ : componentPhiImag_;

        scalarField& out = part == 0 ? Ax.real : Ax.imag;

        // -laplacian(nu,u) + div(phi0,u)
        {
            tmp<volScalarField> tLap = fvc::laplacian(nu_, u);
            tmp<volScalarField> tConv = fvc::div(base_.phi, u);

            const scalarField& lap = tLap().primitiveField();
            const scalarField& conv = tConv().primitiveField();
            forAll(out, celli)
            {
                out[celli] = -lap[celli] + conv[celli];
            }
        }

        // + div(phi' U0_i(face)) - gamma grad(div(phi'))_i
        //
        // Both terms are surface integrals, so they are assembled into one
        // face field and integrated once.
        {
            multiplySurfaceInto(componentFluxWork_, phiP, U0i);

            if (malGamma_ > SMALL)
            {
                tmp<volScalarField> tDiv = fvc::div(phiP);
                tmp<surfaceScalarField> tI = fvc::interpolate(tDiv());
                tDiv.clear();

                addScaledProductToSurface
                (
                    componentFluxWork_,
                    -malGamma_,
                    Sfi,
                    tI()
                );
            }

            tmp<volScalarField::Internal> tIntegral =
                fvc::surfaceIntegrate(componentFluxWork_);

            const scalarField& si = tIntegral();
            forAll(out, celli) out[celli] += si[celli];
        }

        // Doubled-real complex shift.
        const scalar otherCoeff = part == 0 ? sigmaHatImag : -sigmaHatImag;
        const scalarField& ui = u.primitiveField();
        const scalarField& uo = uOther.primitiveField();
        forAll(out, celli)
        {
            out[celli] -= sigmaHatReal*ui[celli];
            out[celli] += otherCoeff*uo[celli];
        }
    }
}


void ShiftedMALSolver::applyMALOseenComponentPreconditioner
(
    const direction cmpt,
    const ComplexComponentVector& rhs,
    ComplexComponentVector& z
)
{
    // Approximate the doubled-real shifted component operator
    //
    //   [ D_i              +sigmaHatImag I ] [u_r] = [r_r]
    //   [ -sigmaHatImag I   D_i            ] [u_i]   [r_i]
    //
    // where
    //
    //   D_i = A_O,i + A_cross,ii - sigmaHatReal I,
    //
    //   A_O,i = div(phi0,.) - div(nu_i grad(.)),
    //   nu_i  = nu + gamma (Sf_i/|Sf|)^2.
    //
    // D_i^{-1} is approximated by the ordinary OpenFOAM mALOseen_U
    // PBiCGStab/DILU solve.  The imaginary shift is retained through a small
    // number of block Gauss-Seidel sweeps:
    //
    //   D_i u_r = r_r - sigmaHatImag u_i,
    //   D_i u_i = r_i + sigmaHatImag u_r.
    //
    // The per-component operator is assembled once in updateComponentMatrices()
    // and reused here; only its source changes between solves.  Build lazily in
    // case this is reached before updateFrozenBaseFaceVelocity().
    if (!componentMatricesValid_) updateComponentMatrices();

    fvScalarMatrix& UEqn = componentMatrix_[cmpt];

    const scalarField& V = base_.U.mesh().V();
    const scalar sigmaHatImag = jacobianSign_*sigmaImag_;
    const int oldDebug = SolverPerformance<scalar>::debug;

    zeroComplexComponent(z);

    scalarField& psi = malOseenU_.primitiveFieldRef();
    scalarField& source = UEqn.source();

    auto solveOnePart =
    [&](const scalarField& uncoupledRhs,
        const scalarField& coupledSolution,
        const scalar coupledCoefficient,
        scalarField& solution)
    {
        // Deterministic zero initial guess.  Only the internal field is used
        // by the solve: the physical boundary values stay zero for the whole
        // run and the coupled ones are exchanged by the solver itself.
        forAll(psi, celli) psi[celli] = 0.0;

        // Reuse the cached constant operator and reset only its source.  The
        // implicit operators contribute nothing to source_, so setting
        // source_ = V*rhs exactly reproduces "== rhs".  The homogeneous-BC
        // boundary coefficients are constant and are applied (and internally
        // saved/restored) by solve(), so the matrix is safe to reuse.
        if (mag(coupledCoefficient) > SMALL)
        {
            forAll(source, celli)
            {
                scalar m = uncoupledRhs[celli];
                m += coupledCoefficient*coupledSolution[celli];
                source[celli] = V[celli]*m;
            }
        }
        else
        {
            forAll(source, celli)
            {
                source[celli] = V[celli]*uncoupledRhs[celli];
            }
        }

        SolverPerformance<scalar>::debug = 0;
        UEqn.solve();
        SolverPerformance<scalar>::debug = oldDebug;

        forAll(solution, celli) solution[celli] = psi[celli];
    };

    // With no imaginary shift there is no doubled-real coupling to iterate.
    const label nSweeps =
        mag(sigmaHatImag) > SMALL ? malOseenShiftSweeps_ : label(1);

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        // D_i u_r + sigmaHatImag u_i = r_r
        solveOnePart(rhs.real, z.imag, -sigmaHatImag, z.real);

        // D_i u_i - sigmaHatImag u_r = r_i
        solveOnePart(rhs.imag, z.real, sigmaHatImag, z.imag);
    }

    SolverPerformance<scalar>::debug = oldDebug;
}


bool ShiftedMALSolver::solveMALDiagonalComponent
(
    const direction cmpt,
    const ComplexComponentVector& rhs,
    ComplexComponentVector& x,
    label& iterations
)
{
    MALComponentSolver* solverPtr = nullptr;

    if (cmpt == 0)
    {
        solverPtr = &malXSolver_();
    }
    else if (cmpt == 1)
    {
        solverPtr = &malYSolver_();
    }
    else
    {
        solverPtr = &malZSolver_();
    }

    solverPtr->solveComponentGMRES
    (
        rhs,
        x,
        malComponentWorkspace_,
        iterations,
        malVelocityBlockPrintInterval_,
        malReportVelocityBlock_
    );

    const bool converged = solverPtr->lastSolveConverged();

    if (!converged && malRequireVelocityBlockConvergence_)
    {
        FatalErrorInFunction
            << "mAL velocity component U" << label(cmpt)
            << " failed after " << iterations << " iterations"
            << exit(FatalError);
    }

    return converged;
}


void ShiftedMALSolver::solveMALPressurePoisson
(
    const volScalarField& rhs,
    volScalarField& result,
    const bool enforceReference
)
{
    // Apply the pressure-Laplacian inverse required by the shifted mAL Schur
    // approximation using native OpenFOAM GAMG:
    //
    //     laplacian(q) = rhs.
    //
    // fvSolution limits mALPressurePoisson to one GAMG iteration, so this is
    // an approximate L^{-1} action (one multigrid cycle), not a converged
    // pressure solve.
    //
    // The Laplacian is constant, so it is assembled and referenced once and
    // only its source is refreshed here.  Reassembling it on every application
    // (twice per outer FGMRES iteration) was pure overhead.
    const scalarField& V = base_.U.mesh().V();
    scalarField& psi = mALPressurePoisson_.primitiveFieldRef();

    if (!pressureMatrix_.valid())
    {
        forAll(psi, celli) psi[celli] = 0.0;
        mALPressurePoisson_.correctBoundaryConditions();

        pressureMatrix_.set(fvm::laplacian(mALPressurePoisson_).ptr());

        fvScalarMatrix& m = pressureMatrix_();

        // pRefCell_ is the local reference cell on exactly one MPI rank and -1
        // on all other ranks.  fvMatrix::setReference handles this same
        // convention used by the known-good Newton pressure projection; its
        // diagonal contribution is constant and is baked in here, its source
        // contribution is re-applied on every solve.
        //
        // needReference() is a collective, so it must be evaluated on every
        // rank and not behind the rank-local pRefCell_ test.
        const bool needsReference = mALPressurePoisson_.needReference();

        if (needsReference && pRefCell_ >= 0)
        {
            pressureReferenceSource_ = m.diag()[pRefCell_]*pRefValue_;
            m.diag()[pRefCell_] += m.diag()[pRefCell_];
        }
    }

    fvScalarMatrix& pEqn = pressureMatrix_();

    // The solution is reset to zero before every application, so the deferred
    // non-orthogonal correction of the assembled matrix stays zero.
    forAll(psi, celli) psi[celli] = 0.0;

    scalarField& source = pEqn.source();
    const scalarField& r = rhs.primitiveField();
    forAll(source, celli) source[celli] = V[celli]*r[celli];

    if (pRefCell_ >= 0 && pressureReferenceSource_ != 0.0)
    {
        source[pRefCell_] += pressureReferenceSource_;
    }

    // Solver controls are taken from fvSolution/solvers/mALPressurePoisson.
    // With maxIter=1 this performs one native GAMG cycle.  Suppress only the
    // native scalar solver-performance line; the numerical solve is unchanged.
    const int oldScalarSolverDebug = SolverPerformance<scalar>::debug;
    SolverPerformance<scalar>::debug = 0;
    pEqn.solve();
    SolverPerformance<scalar>::debug = oldScalarSolverDebug;

    // The imaginary solve deliberately passes mALPressurePoisson_ itself as
    // result.  Avoid a self-copy in that case.
    if (&result != &mALPressurePoisson_)
    {
        copyVolScalar(result, mALPressurePoisson_);
    }

    if (enforceReference)
    {
        enforceZeroPressureReference(result, pRefCell_);
    }
}


void ShiftedMALSolver::applyMALSchurInverse
(
    const ComplexFlow& rhs,
    ComplexFlow& pressure
)
{
    // Shift-aware mAL Schur approximation in the OpenFOAM strong-form sign:
    //   Sp^{-1} ~= -(gamma+nu) I - sigmaHat L_OF^{-1},
    //   sigmaHat = jacobianSign*sigma.
    zeroVolScalar(pressure.real.p);
    zeroVolScalar(pressure.imag.p);

    const scalar a = malGamma_ + nu_.value();
    axpyVolScalar(pressure.real.p, -a, rhs.real.p);
    axpyVolScalar(pressure.imag.p, -a, rhs.imag.p);

    const scalar sigmaHatReal = jacobianSign_*sigmaReal_;
    const scalar sigmaHatImag = jacobianSign_*sigmaImag_;

    if (mag(sigmaHatReal) > SMALL || mag(sigmaHatImag) > SMALL)
    {
        // Store q_r = L_OF^{-1} rhs_r and q_i = L_OF^{-1} rhs_i.
        //
        // Their pressure gauge is fixed here rather than being left to the
        // single reference enforcement at the end.  Deferring it is exact in
        // real arithmetic but ill-conditioned in floating point: one GAMG
        // cycle from a zero start can carry a mean level far larger than the
        // pressure variation, and cancelling it only after scaling loses
        // several digits of the result.
        solveMALPressurePoisson(rhs.real.p, mALPressurePoissonStore_, true);
        solveMALPressurePoisson(rhs.imag.p, mALPressurePoisson_, true);

        // -sigmaHat*(q_r + i q_i)
        axpyVolScalar(pressure.real.p, -sigmaHatReal, mALPressurePoissonStore_);
        axpyVolScalar(pressure.real.p,  sigmaHatImag, mALPressurePoisson_);
        axpyVolScalar(pressure.imag.p, -sigmaHatReal, mALPressurePoisson_);
        axpyVolScalar(pressure.imag.p, -sigmaHatImag, mALPressurePoissonStore_);
    }

    enforceZeroPressureReference(pressure.real.p, pRefCell_);
    enforceZeroPressureReference(pressure.imag.p, pRefCell_);
}


void ShiftedMALSolver::applyMAL(const ComplexFlow& r, ComplexFlow& z)
{
    // Exact sign factorization:
    //   (s J_N - sigma B) z = r
    //   <=> (J_N - s sigma B) z = s r,
    // where s=jacobianSign.  Only the (U,p) rows of r carry information.
    copyFlowStateRows(malCanonicalRhsWork_, r);
    if (jacobianSign_ < 0.0) scaleFlowStateRows(malCanonicalRhsWork_, -1.0);

    // Consistent augmented RHS: f_gamma = f - gamma grad(g).
    if (malGamma_ > SMALL)
    {
        copyInternalAndCorrect(continuityWork_, malCanonicalRhsWork_.real.p);
        tmp<volVectorField> tGradReal = fvc::grad(continuityWork_);
        addFromTmp(malCanonicalRhsWork_.real.U, tGradReal, -malGamma_);
        tGradReal.clear();

        copyInternalAndCorrect(continuityWork_, malCanonicalRhsWork_.imag.p);
        tmp<volVectorField> tGradImag = fvc::grad(continuityWork_);
        addFromTmp(malCanonicalRhsWork_.imag.U, tGradImag, -malGamma_);
    }

    // Block Gauss-Seidel over Ux, Uy, Uz.  z accumulates the velocity solution
    // in place, so the coupling action can read it directly: the components
    // that have not been solved yet are still exactly zero.
    zeroInternalVelocity(z);
    label iterations = 0;

    for (direction cmpt=0; cmpt<3; ++cmpt)
    {
        if (cmpt == 0)
        {
            copyFlowComponentToCompact
            (
                malCompactRhsWork_,
                malCanonicalRhsWork_,
                cmpt
            );
        }
        else
        {
            // Off-diagonal coupling of the already solved components.
            setHomogeneousVelocityBoundary(z);
            replacePhiWithFluxOfU(z);
            applyMALAugmentedVelocityAction(z, malVelocityActionWork_);

            flowComponentDifferenceToCompact
            (
                malCompactRhsWork_,
                malCanonicalRhsWork_,
                malVelocityActionWork_,
                cmpt
            );
        }

        solveMALDiagonalComponent
        (
            cmpt,
            malCompactRhsWork_,
            malCompactSolutionWork_,
            iterations
        );

        copyCompactToVelocityComponent(z, malCompactSolutionWork_, cmpt);
    }

    setHomogeneousVelocityBoundary(z);
    replacePhiWithFluxOfU(z);

    // Paper uses B=-div.  With our continuity row +div, the lower-block RHS is
    // h = div(u_tilde) - g.
    tmp<volScalarField> tDivReal = fvc::div(z.real.phi);
    copyInternalAndCorrect(malPressureRhsWork_.real.p, tDivReal());
    tDivReal.clear();

    tmp<volScalarField> tDivImag = fvc::div(z.imag.phi);
    copyInternalAndCorrect(malPressureRhsWork_.imag.p, tDivImag());
    tDivImag.clear();

    axpyVolScalar(malPressureRhsWork_.real.p, -1.0, malCanonicalRhsWork_.real.p);
    axpyVolScalar(malPressureRhsWork_.imag.p, -1.0, malCanonicalRhsWork_.imag.p);

    // Writes only the pressure rows of z; its velocity/flux rows are final.
    applyMALSchurInverse(malPressureRhsWork_, z);
}


bool ShiftedMALSolver::applyShiftInvert
(
    const ComplexFlow& q,
    ComplexFlow& result,
    label& iterations,
    const label printInterval,
    const bool report
)
{
    applyB(q, rhsBWork_);

    // Every future Arnoldi OP application must be the same deterministic
    // linear map, therefore force a zero initial guess.
    zeroComplexFlow(result);

    solveGMRES
    (
        result,
        rhsBWork_,
        workspace_,
        iterations,
        true,
        printInterval,
        report,
        true
    );

    return lastSolveConverged();
}

} // namespace matrixFreeStability
} // namespace Foam
